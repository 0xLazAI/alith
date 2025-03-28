use std::sync::Arc;

use alith_core::chat::{
    Completion, Message, Request, ResponseContent, ResponseTokenUsage, ResponseToolCalls,
};
use alith_core::interface::llms::api::openai::completion::{
    ChatChoice, ChatCompletionResponseMessage, CompletionRequestMessage, CompletionUsage,
    FinishReason, OpenAICompletionRequest, OpenAICompletionResponse, Role,
};
use alith_core::interface::requests::completion::tool::{Function, ToolCall};
use alith_core::tool::ToolDefinition;
use anyhow::{Result, anyhow};
use bytes::Bytes;
use chrono::{Timelike, Utc};
use http::{Method, Response, StatusCode};
use http_body_util::{BodyExt, Full, combinators::BoxBody};
use hyper::{body::Incoming, service::service_fn};
use hyper_util::rt::{TokioExecutor, TokioIo};
use serde_json::{Value, json};
use std::convert::Infallible;
use tokio::{
    net::TcpListener,
    sync::{RwLock, oneshot},
};
use tokio_graceful::Shutdown;

pub const DEFAULT_ADDR: &str = "127.0.0.1:8080";
pub const V1_CHAT_COMPLETIONS: &str = "/v1/chat/completions";

/// Run an inference server with the given model and address.
pub async fn run(
    addr: Option<String>,
    model: impl Completion + Send + Sync + 'static,
) -> Result<()> {
    let addr = addr.unwrap_or(DEFAULT_ADDR.to_string());
    let server = Arc::new(Server {
        model: RwLock::new(model),
    });
    let listener = TcpListener::bind(&addr).await?;
    let stop_server = server.run(listener).await?;
    shutdown_signal().await?;
    let _ = stop_server.send(());
    Ok(())
}

struct Server<M: Completion + Send + Sync + 'static> {
    model: RwLock<M>,
}

impl<M: Completion + Send + Sync> Server<M> {
    async fn run(self: Arc<Self>, listener: TcpListener) -> Result<oneshot::Sender<()>> {
        let (tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let shutdown = Shutdown::new(async { rx.await.unwrap_or_default() });
            let guard = shutdown.guard_weak();

            loop {
                tokio::select! {
                    res = listener.accept() => {
                        let Ok((cnx, _)) = res else {
                            continue;
                        };

                        let stream = TokioIo::new(cnx);
                        let server = self.clone();
                        shutdown.spawn_task(async move {
                            let hyper_service = service_fn(move |request: hyper::Request<Incoming>| {
                                server.clone().handle(request)
                            });
                            let _ = hyper_util::server::conn::auto::Builder::new(TokioExecutor::new())
                                .serve_connection_with_upgrades(stream, hyper_service).await;
                        });
                    }
                    _ = guard.cancelled() => {
                        break;
                    }
                }
            }
        });
        Ok(tx)
    }

    async fn handle(
        self: Arc<Self>,
        req: hyper::Request<Incoming>,
    ) -> std::result::Result<Response<BoxBody<Bytes, Infallible>>, hyper::Error> {
        let method = req.method().clone();
        let uri = req.uri().clone();
        let path = uri.path();

        if method == Method::OPTIONS {
            let mut res = Response::default();
            *res.status_mut() = StatusCode::NO_CONTENT;
            set_access_control_headers(&mut res);
            return Ok(res);
        }

        let mut status = StatusCode::OK;
        let res = if path == V1_CHAT_COMPLETIONS {
            self.chat_completions(req).await
        } else {
            // TODO: other APIs
            status = StatusCode::NOT_FOUND;
            Err(anyhow!("Not Found"))
        };
        let mut res = match res {
            Ok(res) => res,
            Err(err) => {
                if status == StatusCode::OK {
                    status = StatusCode::BAD_REQUEST;
                }
                ret_err(err)
            }
        };
        *res.status_mut() = status;
        set_access_control_headers(&mut res);
        Ok(res)
    }

    async fn chat_completions(
        &self,
        req: hyper::Request<Incoming>,
    ) -> Result<Response<BoxBody<Bytes, Infallible>>> {
        let req_body = req.collect().await?.to_bytes();
        let req_body: Value = serde_json::from_slice(&req_body)
            .map_err(|err| anyhow!("Invalid request json, {err}"))?;

        let req_body: OpenAICompletionRequest = serde_json::from_value(req_body)
            .map_err(|err| anyhow!("Invalid request body, {err}"))?;

        let OpenAICompletionRequest {
            model,
            messages,
            temperature,
            top_p,
            max_tokens,
            tools,
            ..
        } = req_body;

        let id = generate_completion_id();
        let created = Utc::now().timestamp();
        let result = self
            .model
            .write()
            .await
            .completion(Request {
                prompt: "".to_string(),
                preamble: "".to_string(),
                history: unsafe {
                    std::mem::transmute::<Vec<CompletionRequestMessage>, Vec<Message>>(
                        messages.clone(),
                    )
                },
                max_tokens,
                temperature,
                top_p,
                tools: tools
                    .unwrap_or_default()
                    .iter()
                    .map(|tool| ToolDefinition {
                        name: tool.function.name.clone(),
                        description: tool.function.description.clone(),
                        parameters: tool.function.parameters.clone(),
                    })
                    .collect(),
                ..Default::default()
            })
            .await?;
        let toolcalls = result.toolcalls();
        let usage = result.token_usage();
        let choice = ChatChoice {
            index: 0,
            message: ChatCompletionResponseMessage {
                content: Some(result.content()),
                role: Role::Assistant,
                tool_calls: if toolcalls.is_empty() {
                    None
                } else {
                    Some(
                        toolcalls
                            .iter()
                            .map(|tool| ToolCall {
                                id: tool.id.clone(),
                                r#type: tool.r#type.clone(),
                                function: Function {
                                    name: tool.function.name.clone(),
                                    arguments: tool.function.arguments.clone(),
                                },
                            })
                            .collect(),
                    )
                },
            },
            finish_reason: Some(if toolcalls.is_empty() {
                FinishReason::Stop
            } else {
                FinishReason::ToolCalls
            }),
            logprobs: None,
        };
        let resp = OpenAICompletionResponse {
            id,
            choices: vec![choice],
            created: created as u32,
            model,
            usage: Some(CompletionUsage {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
            }),
        };
        let bytes = Bytes::from(
            serde_json::to_string(&resp)
                .map_err(|err| anyhow!("Failed to serialize response, {err}"))?,
        );
        let res = Response::builder()
            .header("Content-Type", "application/json")
            .body(Full::new(bytes).boxed())?;
        Ok(res)
    }
}

#[inline]
fn generate_completion_id() -> String {
    format!("chat-{}", Utc::now().nanosecond())
}

#[inline]
fn set_access_control_headers(res: &mut Response<BoxBody<Bytes, Infallible>>) {
    res.headers_mut().insert(
        hyper::header::ACCESS_CONTROL_ALLOW_ORIGIN,
        hyper::header::HeaderValue::from_static("*"),
    );
    res.headers_mut().insert(
        hyper::header::ACCESS_CONTROL_ALLOW_METHODS,
        hyper::header::HeaderValue::from_static("GET,POST,PUT,PATCH,DELETE"),
    );
    res.headers_mut().insert(
        hyper::header::ACCESS_CONTROL_ALLOW_HEADERS,
        hyper::header::HeaderValue::from_static("Content-Type,Authorization"),
    );
}

#[inline]
fn ret_err<T: std::fmt::Display>(err: T) -> Response<BoxBody<Bytes, Infallible>> {
    let data = json!({
        "error": {
            "message": err.to_string(),
            "type": "invalid_request_error",
        },
    });
    Response::builder()
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(data.to_string())).boxed())
        .unwrap()
}

async fn shutdown_signal() -> Result<()> {
    tokio::signal::ctrl_c().await?;
    Ok(())
}
