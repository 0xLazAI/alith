use alith::{LLM, client::prelude::*};

/// Enforces CoT style reasoning on the output of an LLM, before returning the requested primitive. Currently, reason is bound to the one_round reasoning workflow. Workflows relying on grammars are only supported by local LLMs.
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::from_model_name("gpt-4")?;
    let client = model.client();
    client.chat_completion().max_tokens(90);
    // A boolean reason request
    let response = client
        .reason()
        .boolean()
        .set_instructions("Is the sky blue?")
        .return_primitive()
        .await
        .unwrap();
    assert!(response);
    // An integer reason request
    let mut reason_request = client.reason().integer();
    // Settings specific to the primitive can be accessed through the primitive field
    reason_request.primitive.lower_bound(0).upper_bound(10);
    let response = reason_request
        .set_instructions("How many fingers do you have?")
        .return_primitive()
        .await
        .unwrap();
    assert_eq!(response, 5);

    // Options
    let mut reason_request = client.reason().integer();
    // The conclusion and reasoning sentences can be set. This is useful for more complex reasoning tasks where you want the llm to pontificate more.
    reason_request
        .conclusion_sentences(4)
        .reasoning_sentences(3);

    // An integer request, but with an optional response
    let response = reason_request
        .set_instructions("How many coins are in my pocket?")
        .return_optional_primitive()
        .await
        .unwrap();
    assert_eq!(response, None);

    // An exact string reason request
    let mut reason_request = client.reason().exact_string();
    reason_request.primitive.add_string_to_allowed("red");
    reason_request
        .primitive
        .add_strings_to_allowed(&["blue", "green"]);

    let response = reason_request
        .set_instructions("What color is clorophyll?")
        .return_primitive()
        .await
        .unwrap();
    println!("{response}");
    Ok(())
}
