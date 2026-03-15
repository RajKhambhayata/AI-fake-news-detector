import csv
import os

FAKE_DATA = [
    ["Scientists prove earth is flat", "A recent study out of an unknown institute has finally proven that the earth is actually a flat disc floating in space."],
    ["New 6G network causes flu", "Local authorities are warning citizens that the new 6G towers are directly causing the latest flu outbreak."],
    ["Aliens landed in New York", "Scores of people reported seeing a flying saucer crash into Central Park yesterday evening."],
    ["Drinking bleach cures all diseases", "A shocking new health trend claims that drinking household bleach will immediately cure any virus or bacteria."],
    ["Moon landing was filmed in a basement", "Declassified documents reveal that the 1969 moon landing was entirely filmed in a Hollywood basement."],
]

TRUE_DATA = [
    ["NASA launches new rover to Mars", "The space agency successfully launched its latest rover mission to explore the Martian surface for signs of ancient life."],
    ["Global markets rally after rate cut", "Stock markets around the world saw significant gains today following the central bank's decision to cut interest rates."],
    ["New cancer treatment shows promise", "Clinical trials for a novel immunotherapy drug have shown promising results in treating advanced stages of melanoma."],
    ["Tech giant announces quarterly profits", "The leading technology company reported a 15% increase in revenue for the third quarter, exceeding analyst expectations."],
    ["Local team wins championship", "The city's football team secured their first championship title in over two decades after a thrilling final match."],
]

def main():
    os.makedirs('e:/AI/fakenews_project/scripts', exist_ok=True)
    
    with open('e:/AI/fakenews_project/scripts/Fake.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'text', 'subject', 'date'])
        for row in FAKE_DATA:
            writer.writerow([row[0], row[1], 'News', '2023-01-01'])
            
    with open('e:/AI/fakenews_project/scripts/True.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'text', 'subject', 'date'])
        for row in TRUE_DATA:
            writer.writerow([row[0], row[1], 'News', '2023-01-01'])
            
    print("Created mock Fake.csv and True.csv files.")

if __name__ == "__main__":
    main()
