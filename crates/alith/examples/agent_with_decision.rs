use alith::{
    LLM,
    client::{DecisionTrait, prelude::*},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::from_model_name("gpt-4")?;
    let client = model.client();
    // A boolean decision request
    let response = client
        .reason()
        .boolean()
        .decision()
        .set_instructions("Is the sky blue?")
        .return_primitive()
        .await
        .unwrap();
    assert!(response);

    // An integer decision request
    let mut reason_request = client.reason().integer();
    // Settings specific to the primitive can be accessed through the primitive field
    reason_request.primitive.lower_bound(0).upper_bound(10);
    let mut decision_request = reason_request.decision();
    let response = decision_request
        .set_instructions("How many fingers do you have?")
        .return_primitive()
        .await
        .unwrap();
    assert_eq!(response, 5);

    // Options
    let mut decision_request = client.reason().integer().decision();
    // Set the number of 'votes', or rounds of reasoning, to be conducted
    decision_request.best_of_n_votes(5);
    // Uses a temperature gradient for each round of reasoning
    decision_request.dynamic_temperature(true);

    // An integer request, but with an optional response
    let response = decision_request
        .set_instructions("How many coins are in my pocket?")
        .return_optional_primitive()
        .await
        .unwrap();
    assert_eq!(response, None);
    Ok(())
}
