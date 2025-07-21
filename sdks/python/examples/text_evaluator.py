from alith.data.evaluator.text import TextEvaluator
from alith.lazai.node.evaluator import StandardDQSCalculator

evaluator = TextEvaluator()
print(evaluator.evaluate_accuracy("Hello, what's your name?"))
print(evaluator.evaluate_accuracy("Helle, what's your name?"))
print(evaluator.evaluate_accuracy("123123,sa aosdhaosdh"))
print(evaluator.evaluate_similarity("Hello world", "Hello world"))

calulator = StandardDQSCalculator()
print(int(calulator.calculate("Hello, what's your name?", "text") * 10000))
