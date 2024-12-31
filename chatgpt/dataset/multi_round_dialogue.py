import json
from typing import Dict, List

from tqdm import tqdm


class UltraChatProcessor(object):

    def __init__(self):
        super.__init__()

    def get_examples(self, data_path: str) -> List[Dict]:
        examples = []
        j = 0
        with open(data_path) as f:
            for line in tqdm(f.readlines()):
                if line.strip():
                    data = json.loads(line)
                    dialogue = data['data']
                    tags = [
                        i for _ in range(len(dialogue) // 2)
                        for i in ['Human', 'Assistant']
                    ]
                    example = {
                        'id': 'identity_' + str(j),
                        'conversations': [],
                    }
                    for i in range(0, len(dialogue), 2):
                        tgt_text = dialogue[i + 1]
                        context = dialogue[:i + 1]
                        human_content = {
                            'from': tags[:i + 1],
                            'value': context,
                        }
                        bot_content = {'from': tags[i + 1], 'value': tgt_text}
                        example['conversations'].append(human_content)
                        example['conversations'].append(bot_content)

                    examples.append(example)
                    j += 1
        return examples

    def dump(self, data):
        with open('data.json', 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == '__main__':

    path = ''
