//! Reference: https://dagrs.com/docs/getting-started/components
pub use dagrs::{
    auto_node, dependencies, Action, Content, DefaultNode, EmptyAction, EnvVar, Graph, InChannels,
    Node, NodeId, NodeName, NodeTable, OutChannels, Output, RecvErr, SendErr,
};

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use dagrs::*;

    #[derive(Default)]
    pub struct HelloAction;

    #[async_trait]
    impl Action for HelloAction {
        async fn run(&self, _: &mut InChannels, _: &OutChannels, _: Arc<EnvVar>) -> Output {
            Output::Out(Some(Content::new("Hello world".to_string())))
        }
    }

    #[tokio::test]
    async fn node_test() {
        let node_name = "My Node";
        let mut node_table = NodeTable::new();
        let mut node =
            DefaultNode::with_action(NodeName::from(node_name), HelloAction, &mut node_table);

        assert_eq!(node_table.get(node_name).unwrap(), &node.id());

        let env = Arc::new(EnvVar::new(node_table));
        let out = node.run(env).await.get_out().unwrap();
        let out: &String = out.get().unwrap();
        assert_eq!(out, "Hello world");
    }

    #[test]
    fn graph_test() {
        let mut node_table = NodeTable::new();
        let s = DefaultNode::with_action(NodeName::from("s"), HelloAction, &mut node_table);
        let a = DefaultNode::with_action(NodeName::from("a"), HelloAction, &mut node_table);
        let b = DefaultNode::with_action(NodeName::from("b"), HelloAction, &mut node_table);
        let mut g = dependencies!(
            s -> a b,
            b -> a
        );
        g.start();
    }
}
