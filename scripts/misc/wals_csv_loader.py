import csv
from collections import defaultdict


class WalsCsvLoader:
    def __init__(self):
        pass

    def load(self, language_file_path):
        languages = []
        feature_values = defaultdict(set)

        with open(language_file_path) as fw:
            reader = csv.reader(fw)
            header_line = next(reader)

            for l in reader:
                # exclude Sign languages and Creoles
                if l[6] == 'Sign Languages' or l[6] == 'Creoles and Pidgins':
                    continue

                language = {}

                language['id'] = l[0]
                language['name'] = l[3]
                language['longitude'] = float(l[4])
                language['latitude'] = float(l[5])
                language['genus'] = l[6]
                language['family'] = l[7]
                language['glottocode'] = l[2]
                language['iso_code'] = l[1]

                language['features'] = {}

                for feature_idx in range(10, len(header_line)):
                    feature_name = header_line[feature_idx]
                    if not l[feature_idx] == '':
                        feature_values[feature_name].add(l[feature_idx])
                    language['features'][feature_name] = 'NA' if l[feature_idx] == '' else l[feature_idx]

                languages.append(language)

        return {'languages': languages,
                'feature_values': feature_values}
