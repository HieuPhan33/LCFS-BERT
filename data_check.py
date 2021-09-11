import xml.etree.ElementTree as ET
domains = ['Laptops', 'Restaurants']
for domain in domains:
    aspects = []
    with open(f'datasets/semeval14/{domain}_Test_Gold.xml.seg') as f:
       for i, line in enumerate(f):
           if i % 3 == 1:
               aspects.append(line.strip())
    aspects_xml = []
    conflict_terms = []
    positive_terms = []
    negative_terms = []
    neutral_terms = []
    root = ET.parse(f'semeval14/{domain}_Test_Gold.xml').getroot()
    for tag in root.findall('.//aspectTerm'):
        label = tag.attrib['polarity']
        term = tag.attrib['term']
        if label == 'conflict':
            conflict_terms.append(term)
        elif label == 'positive':
            positive_terms.append(term)
        elif label == 'negative':
            negative_terms.append(term)
        elif label == 'neutral':
            neutral_terms.append(term)
        aspects_xml.append(term)
    print(f"Statistics for test files in domain: {domain}")
    print(f"Positive {len(positive_terms)} || Negative {len(negative_terms)} || Neutral {len(neutral_terms)}")
    print("Difference between original and preprocessed terms:",len(aspects_xml) - len(aspects))
    print("Number of conflict terms :", len(conflict_terms))
    print("===================")