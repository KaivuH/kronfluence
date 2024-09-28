import random
from typing import List, Dict, Any, Tuple

class LogicalReasoningDataset:
    def __init__(self, n_vars: int, m_rules: int, frac_train: float, seed: int = 42):
        random.seed(seed)
        self.n_vars = n_vars
        self.m_rules = m_rules
        self.frac_train = frac_train

        # Generate variables and assign random truth values
        self.variables = [f'V{i:03d}' for i in range(n_vars)]
        self.variable_values = {var: random.choice([True, False]) for var in self.variables}

        # Generate rule names
        self.rule_names = [f'R{i:04d}' for i in range(m_rules)]

        # Generate random logical expressions for rules
        self.rules = {}
        self.max_expr_depth = 5
        for rule_name in self.rule_names:
            expr = self.generate_random_expression(self.max_expr_depth)
            self.rules[rule_name] = expr

        # Build vocabulary
        tokens = set(self.variables + self.rule_names + ['AND', 'OR', 'NOT', 'DEFINED_AS', 'EVAL', 'True', 'False', '(', ')'])
        self.idx2vocab = sorted(tokens)
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.idx2vocab)}
        self.n_vocab = len(self.idx2vocab)

        # Split rules into train and validation sets for evaluations
        random.shuffle(self.rule_names)
        n_train_eval = int(len(self.rule_names) * frac_train)
        self.train_eval_rules = self.rule_names[:n_train_eval]
        self.val_eval_rules = self.rule_names[n_train_eval:]

        # Create rule definitions
        self.rule_definitions = []
        for rule_name in self.rule_names:
            expr_tokens = self.expression_to_tokens(self.rules[rule_name])
            definition = [rule_name, 'DEFINED_AS'] + expr_tokens
            self.rule_definitions.append(definition)

        # Create training examples
        self.train_examples = []
        for definition in self.rule_definitions:
            self.train_examples.append((definition, []))  # No output for definitions

        for eval_query in self.train_eval_rules:
            # Evaluate the rule
            result = self.evaluate_expression(eval_query, self.variable_values)
            result_token = 'True' if result else 'False'
            self.train_examples.append(([eval_query, 'EVAL'], [result_token]))

        # Create validation examples
        self.val_examples = []
        for eval_query in self.val_eval_rules:
            result = self.evaluate_expression(eval_query, self.variable_values)
            result_token = 'True' if result else 'False'
            self.val_examples.append(([eval_query, 'EVAL'], [result_token]))

    def generate_random_expression(self, max_depth: int):
        # Decrease the early stopping probability to bias towards longer expressions
        stop_probability = 0.2  # Decreased from 0.3 to 0.1
        if max_depth == 0 or random.random() < stop_probability:
            # Return a variable
            return random.choice(self.variables)
        else:
            op = random.choice(['AND', 'OR', 'NOT'])
            if op == 'NOT':
                expr = self.generate_random_expression(max_depth - 1)
                return ['NOT', expr]
            else:
                left = self.generate_random_expression(max_depth - 1)
                right = self.generate_random_expression(max_depth - 1)
                return [op, left, right]

    def expression_to_tokens(self, expr):
        if isinstance(expr, str):
            return [expr]
        elif isinstance(expr, list):
            if expr[0] == 'NOT':
                return ['NOT', '('] + self.expression_to_tokens(expr[1]) + [')']
            elif expr[0] in ['AND', 'OR']:
                return ['('] + self.expression_to_tokens(expr[1]) + [')'] + [expr[0]] + ['('] + self.expression_to_tokens(expr[2]) + [')']
            else:
                raise ValueError(f"Unknown operator: {expr[0]}")
        else:
            raise ValueError(f"Invalid expression: {expr}")

    def evaluate_expression(self, expr, variable_values):
        if isinstance(expr, str):
            # It's a variable or rule name
            if expr in variable_values:
                return variable_values[expr]
            elif expr in self.rules:
                rule_expr = self.rules[expr]
                return self.evaluate_expression(rule_expr, variable_values)
            else:
                raise ValueError(f"Unknown variable or rule: {expr}")
        elif isinstance(expr, list):
            if expr[0] == 'NOT':
                val = self.evaluate_expression(expr[1], variable_values)
                return not val
            elif expr[0] == 'AND':
                val1 = self.evaluate_expression(expr[1], variable_values)
                val2 = self.evaluate_expression(expr[2], variable_values)
                return val1 and val2
            elif expr[0] == 'OR':
                val1 = self.evaluate_expression(expr[1], variable_values)
                val2 = self.evaluate_expression(expr[2], variable_values)
                return val1 or val2
            else:
                raise ValueError(f"Unknown operator: {expr[0]}")
        else:
            raise ValueError(f"Invalid expression: {expr}")

    def encode(self, sequence):
        return [self.vocab2idx[token] for token in sequence]

    def decode(self, sequence):
        return [self.idx2vocab[idx] for idx in sequence]

    def fetch_train_example(self):
        example = random.choice(self.train_examples)
        input_seq = self.encode(example[0])
        output_seq = self.encode(example[1]) if example[1] else []
        return input_seq, output_seq, example

    def fetch_val_example(self):
        example = random.choice(self.val_examples)
        input_seq = self.encode(example[0])
        output_seq = self.encode(example[1])
        return input_seq, output_seq, example

# Example usage:
# Existing imports and class ...

if __name__ == "__main__":
    dataset = LogicalReasoningDataset(n_vars=100, m_rules=1000, frac_train=0.7, seed=42)
    print(f"Number of variables: {len(dataset.variables)}")
    print(f"Number of rules: {len(dataset.rule_names)}")
    
    print("\nSample Variables and their truth values:")
    for var, value in list(dataset.variable_values.items())[:5]:
        print(f"{var}: {value}")
    
    print("\nSample Rule Definitions:")
    for definition in dataset.rule_definitions[:5]:
        print(' '.join(definition))
    
    print("\nSample Training Evaluation Queries:")
    for eval_query in dataset.train_eval_rules[:5]:
        input_seq = [eval_query, 'EVAL']
        result = dataset.evaluate_expression(eval_query, dataset.variable_values)
        result_token = 'True' if result else 'False'
        print(f"Input: {' '.join(input_seq)}")
        print(f"Output: {result_token}")
    
    print("\nSample Validation Examples:")
    for _ in range(3):
        input_seq, output_seq, example = dataset.fetch_val_example()
        print(f"Input: {' '.join(dataset.decode(input_seq))}")
        print(f"Expected Output: {' '.join(dataset.decode(output_seq))}")