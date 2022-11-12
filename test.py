import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

class_names = ["False", "True"]


def test():
    def print_my_examples(inputs, results):
        result_for_printing = \
            [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f} ({class_names[int(tf.round(results[i][0]))]})'
             for i in range(len(inputs))]
        print(*result_for_printing, sep='\n')
        print()

    examples = [
        'The CIA killed president John F. Kennedy.',
        'COVID-19 originated in Wuhan, China.',
        'COVID-19 was made in a Ukrainian lab, and that is why Putin is invading',
        'Donald Trump won the 2020 election.',
        'Dominion voting machines are controlled by Hillary Clinton.',
        'Aliens built the pyramids in Egypt.',
        'Ukraine has bioweapon labs.',
        'President Bush orchestrated the September 11 attacks.'
    ]

    reloaded_model = tf.saved_model.load(f'./fact_or_fiction_bert')

    reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))

    print('Results from the saved model:')
    print_my_examples(examples, reloaded_results)


test()
