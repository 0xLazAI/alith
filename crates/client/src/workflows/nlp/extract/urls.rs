use crate::{
    components::{
        cascade::{CascadeFlow, step::StepConfig},
        instruct_prompt::{InstructPrompt, InstructPromptTrait},
    },
    primitives::*,
};
use alith_interface::requests::{
    completion::CompletionRequest,
    req_components::{RequestConfig, RequestConfigTrait},
};
use anyhow::Result;
use linkify::{LinkFinder, LinkKind};
use std::{collections::HashSet, str::FromStr};
use url::Url;

pub fn extract_urls<T: AsRef<str>>(input: T) -> Vec<Url> {
    let mut unique_urls = HashSet::new();

    LinkFinder::new()
        .kinds(&[LinkKind::Url])
        .links(input.as_ref())
        .filter_map(|link| Url::from_str(link.as_str()).ok())
        .filter(|url| unique_urls.insert(url.clone()))
        .collect()
}

#[derive(Clone)]
pub struct ExtractUrls {
    pub base_req: CompletionRequest,
    pub instruct_prompt: InstructPrompt,
    pub criteria: Option<String>,
    pub results: Vec<String>,
    pub max_urls: usize,
}

impl ExtractUrls {
    pub fn new(base_req: CompletionRequest) -> Self {
        ExtractUrls {
            instruct_prompt: InstructPrompt::new(),
            base_req,
            criteria: None,
            results: Vec::new(),
            max_urls: 5,
        }
    }

    pub fn max_urls(mut self, max_urls: usize) -> Self {
        self.max_urls = max_urls;
        self
    }

    pub async fn run_return_urls(&mut self) -> Result<Option<Vec<Url>>> {
        Ok(self.run_return_result().await?.results)
    }

    pub async fn run_return_result(&mut self) -> Result<ExtractUrlResult> {
        let flow = self.run_backend().await?;
        if self.results.is_empty() {
            Ok(ExtractUrlResult::new(
                flow,
                None,
                self.criteria.as_ref().unwrap(),
            ))
        } else {
            Ok(ExtractUrlResult::new(
                flow,
                Some(
                    self.results
                        .iter()
                        .map(|url| Url::parse(url).unwrap())
                        .collect(),
                ),
                self.criteria.as_ref().unwrap(),
            ))
        }
    }

    async fn run_backend(&mut self) -> Result<CascadeFlow> {
        let mut primitive = ExactStringPrimitive::default();

        let mut urls_from_instructions: Vec<Url> = Vec::new();
        if let Some(instructions) = self.instruct_prompt.build_instructions() {
            urls_from_instructions.extend(extract_urls(instructions));
        }
        if let Some(supporting_material) = self.instruct_prompt.build_supporting_material() {
            urls_from_instructions.extend(extract_urls(supporting_material));
        }
        if urls_from_instructions.is_empty() {
            return Err(anyhow::anyhow!("No URLs found in the instructions"));
        }

        primitive.add_strings_to_allowed(&urls_from_instructions);

        let mut flow = self.set_criteria().await?;

        self.run_cascade(&mut flow, &mut primitive).await?;
        Ok(flow)
    }

    async fn set_criteria(&mut self) -> Result<CascadeFlow> {
        let mut flow = CascadeFlow::new("ExtractUrls");
        flow.open_cascade();
        flow.new_round(
            "We are extracting URLs from text. Please provide examples of extracting URLs with the instructions: 'Which of these URLs are commonly used in webdev tutorials?'").add_guidance_step(
            &StepConfig::default(),
            "`https://www.example.com is commonly used in webdev tutorials: true.` In this example, the URL satisfies the criteria: 'is commonly used in webdev tutorials.' Therefore, the URL should be extracted from the text.\n`https://www.zombo.com is commonly used in webdev tutorials: false.`. In this example, the URL does not satisfy the criteria: 'is commonly used in webdev tutorials.' Therefore, the URL should not be extracted from the text.",
        );
        flow.last_round()?.run_all_steps(&mut self.base_req).await?;

        let initial_qualities_task = format!(
            "We are extracting URLs from text using the instructions:\n{} Briefly describe the criteria of the URLs to be extracted.",
            self.instruct_prompt.build_instructions().unwrap()
        );
        let config = StepConfig {
            step_prefix: Some("Criteria: ".to_owned()),
            grammar: TextPrimitive::default().text_token_length(200).grammar(),
            ..StepConfig::default()
        };
        flow.new_round(initial_qualities_task)
            .add_inference_step(&config);
        flow.last_round()?.run_all_steps(&mut self.base_req).await?;

        let refine_criteria_task = format!(
            "Reframe the instructions and criteria into a statment used to evaluate if a URL should be extracted. This statement should have a boolean answer. The answer should represent whether or not the URL satisfies the criteria. This should be a single sentence 'is' statment; as in, 'The URL is <criteria>: true or false'.\nCriteria:\n{}\nInstructions:\n{}",
            flow.primitive_result().unwrap(),
            self.instruct_prompt.build_instructions().unwrap()
        );
        let config = StepConfig {
            step_prefix: Some("The URL is ".to_owned()),
            stop_word_done: ": true or false".to_owned(),
            grammar: TextPrimitive::default().text_token_length(200).grammar(),
            ..StepConfig::default()
        };
        flow.new_round(refine_criteria_task)
            .add_inference_step(&config);
        flow.last_round()?.run_all_steps(&mut self.base_req).await?;
        self.criteria = Some(flow.primitive_result().unwrap());
        Ok(flow)
    }

    async fn extract_step(
        &mut self,
        flow: &mut CascadeFlow,
        primitive: &mut ExactStringPrimitive,
    ) -> Result<()> {
        let config = StepConfig {
            cache_prompt: true,
            stop_word_no_result: Some("No qualifying URLs.".to_owned()),
            grammar: primitive.grammar(),
            ..StepConfig::default()
        };

        flow.last_round()?.add_inference_step(&config);
        flow.last_round()?.run_next_step(&mut self.base_req).await
    }

    async fn validate_step(&mut self, flow: &mut CascadeFlow) -> Result<bool> {
        let config = StepConfig {
            cache_prompt: true,
            step_prefix: Some(format!(" is {}: ", self.criteria.as_ref().unwrap())),
            grammar: BooleanPrimitive::default().grammar(),
            ..StepConfig::default()
        };

        flow.last_round()?.add_inference_step(&config);
        flow.last_round()?.run_next_step(&mut self.base_req).await?;
        if flow.primitive_result().unwrap().parse().unwrap() {
            Ok(true)
        } else {
            flow.last_round()?
                .last_step()?
                .set_dynamic_suffix(". I apologize. This URL does not meet the criteria and was returned by mistake. In the future, we'll only return URLs that satisfy the criteria.\n".to_owned());
            Ok(false)
        }
    }

    async fn check_for_remaining(
        &mut self,
        flow: &mut CascadeFlow,
        primitive: &mut ExactStringPrimitive,
    ) -> Result<bool> {
        let remaining_urls = primitive.allowed_strings.join(", ");
        let config = StepConfig {
            cache_prompt: true,
            step_prefix: Some(format!(
                "At least one of the remaining URLs, {remaining_urls}, is {}: ",
                self.criteria.as_ref().unwrap()
            )),
            grammar: BooleanPrimitive::default().grammar(),
            ..StepConfig::default()
        };

        flow.last_round()?.add_inference_step(&config);
        flow.last_round()?.run_next_step(&mut self.base_req).await?;
        flow.last_round()?
            .last_step()?
            .set_dynamic_suffix(".\n".to_owned());
        if flow.primitive_result().unwrap().parse().unwrap() {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn run_cascade(
        &mut self,
        flow: &mut CascadeFlow,
        primitive: &mut ExactStringPrimitive,
    ) -> Result<()> {
        let task = format!(
            "Text with URLs to extract:\n{}\nReturn the URL that is most likely relevant to the criteria. If you are certain the text contains no qualifying URLs say 'No qualifying URLs.'.\nCriteria:\n This URL is {}.",
            self.instruct_prompt.build_supporting_material().unwrap(),
            self.criteria.as_ref().unwrap()
        );
        flow.new_round(task).step_separator = None;
        flow.last_round()?.open_round(&mut self.base_req)?;
        for i in 1..=primitive.allowed_strings.len() {
            if self.results.len() >= self.max_urls {
                break;
            }
            if i > 1 {
                flow.new_round("Return the next URL that is likely to satisfy the criteria, or if there are no more URLs to extract say 'No qualifying URLs.'.").step_separator = None;
                flow.last_round()?.open_round(&mut self.base_req)?;
            }
            self.extract_step(flow, primitive).await?;
            match flow.primitive_result() {
                Some(url_result) => {
                    primitive.remove_string_from_allowed(&url_result);
                    if self.validate_step(flow).await? {
                        self.results.push(url_result);
                    } else if !self.check_for_remaining(flow, primitive).await? {
                        flow.last_round()?.close_round(&mut self.base_req)?;
                        break;
                    }
                    flow.last_round()?.close_round(&mut self.base_req)?;
                }
                None => {
                    flow.last_round()?.close_round(&mut self.base_req)?;
                    break;
                }
            };
        }

        flow.close_cascade()?;
        Ok(())
    }
}

impl RequestConfigTrait for ExtractUrls {
    fn config(&mut self) -> &mut RequestConfig {
        &mut self.base_req.config
    }

    fn reset_request(&mut self) {
        self.instruct_prompt.reset_instruct_prompt();
        self.base_req.reset_completion_request();
    }
}

impl InstructPromptTrait for ExtractUrls {
    fn instruct_prompt_mut(&mut self) -> &mut InstructPrompt {
        &mut self.instruct_prompt
    }
}

#[derive(Clone)]
pub struct ExtractUrlResult {
    pub results: Option<Vec<Url>>,
    pub criteria: String,
    pub duration: std::time::Duration,
    pub workflow: CascadeFlow,
}

impl ExtractUrlResult {
    fn new(flow: CascadeFlow, results: Option<Vec<Url>>, criteria: &str) -> Self {
        ExtractUrlResult {
            results,
            criteria: criteria.to_owned(),
            duration: flow.duration,
            workflow: flow,
        }
    }
}

impl std::fmt::Display for ExtractUrlResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(
            f,
            "\x1b[38;5;45m\x1b[1m{}\x1b[0m",
            self.workflow.cascade_name
        )?;
        writeln!(f)?;
        for (i, round) in self.workflow.rounds.iter().enumerate() {
            writeln!(f, "\x1b[38;5;44mRound {}\x1b[0m", i + 1)?;
            writeln!(f, "{round}",)?;
        }
        writeln!(f, "\x1b[38;5;42mcriteria\x1b[0m: {:?}", self.criteria)?;
        writeln!(f, "\x1b[38;5;43mduration\x1b[0m: {:?}", self.duration)?;
        Ok(())
    }
}
