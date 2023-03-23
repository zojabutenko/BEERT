from flask import Flask, request, render_template
import torch
from transformers import BertTokenizer
import numpy as np
from torch import nn

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


class Dataset(torch.utils.data.Dataset):
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

class BertClassifier(nn.Module):
    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

flat_id2label = {0: 'Pale Lagers. American Adjunct Lager', 1: 'Dark Lagers. Marzen', 2: 'Pale Lagers. German Pilsner', 3: 'Pale Lagers. European Pale Lager', 4: 'Pale Lagers. American Lager', 5: 'Pale Lagers. Helles', 6: 'Stouts. Russian Imperial Stout', 7: 'Stouts. American Imperial Stout', 8: 'Pale Lagers. Bohemian / Czech Pilsner', 9: 'Stouts. Sweet / Milk Stout', 10: 'India Pale Ales. Imperial IPA', 11: 'Strong Ales. Tripel', 12: 'Porters. American Porter', 13: 'Stouts. American Stout', 14: 'India Pale Ales. American IPA', 15: 'Pale Ales. English Pale Ale', 16: 'Specialty Beers. Fruit and Field Beer', 17: 'Stouts. Oatmeal Stout', 18: 'Wheat Beers. Hefeweizen', 19: 'Strong Ales. Belgian Pale Strong Ale', 20: 'Specialty Beers. Pumpkin Beer', 21: 'Strong Ales. Belgian Dark Strong Ale', 22: 'Brown Ales. American Brown Ale', 23: 'Pale Ales. American Blonde Ale', 24: 'India Pale Ales. New England IPA', 25: 'Wheat Beers. Witbier', 26: 'Pale Ales. Saison', 27: 'Pale Ales. American Pale Ale', 28: 'Pale Ales. Kölsch', 29: 'Strong Ales. English Barleywine', 30: 'Pale Ales. American Amber / Red Ale', 31: 'Wheat Beers. American Pale Wheat Beer', 32: 'Porters. Imperial Porter', 33: 'Wild/Sour Beers. Wild Ale', 34: 'Strong Ales. American Barleywine', 35: 'Wild/Sour Beers. Gose', 36: 'Strong Ales. Scotch Ale / Wee Heavy', 37: 'Wild/Sour Beers. Fruited Kettle Sour', 38: 'Pale Lagers. Light Lager', 39: 'Bocks. Doppelbock', 40: 'Dark Ales. Winter Warmer', 41: 'Wild/Sour Beers. Berliner Weisse', 42: 'Strong Ales. Quadrupel (Quad)', 43: 'Pale Lagers. Kellerbier / Zwickelbier', 44: 'Pale Ales. Belgian Pale Ale', 45: 'Porters. Baltic Porter', 46: 'Brown Ales. English Brown Ale', 47: 'Strong Ales. American Strong Ale', 48: 'India Pale Ales. English IPA', 49: 'Wild/Sour Beers. Fruit Lambic', 50: 'Dark Lagers. Munich Dunkel', 51: 'Specialty Beers. Low-Alcohol Beer', 52: 'India Pale Ales. Black IPA', 53: 'Pale Ales. English Bitter', 54: 'Dark Lagers. Schwarzbier', 55: 'Porters. English Porter', 56: 'Dark Lagers. Vienna Lager', 57: 'Bocks. Bock', 58: 'Pale Ales. Extra Special / Strong Bitter (ESB)', 59: 'Specialty Beers. Herb and Spice Beer', 60: 'India Pale Ales. Milkshake IPA', 61: 'Bocks. Maibock', 62: 'Dark Ales. Dubbel', 63: 'Pale Ales. Irish Red Ale', 64: 'Stouts. Irish Dry Stout', 65: 'Dark Lagers. American Amber / Red Lager', 66: 'Brown Ales. Altbier', 67: 'India Pale Ales. Belgian IPA', 68: 'Dark Ales. Scottish Ale', 69: 'Specialty Beers. Rye Beer', 70: 'Pale Lagers. European / Dortmunder Export Lager', 71: 'Pale Lagers. Festbier / Wiesnbier', 72: 'Stouts. Foreign / Export Stout', 73: 'Pale Lagers. European Strong Lager', 74: 'Pale Ales. Belgian Blonde Ale', 75: 'Dark Lagers. European Dark Lager', 76: 'Stouts. English Stout', 77: 'Pale Ales. Bière de Garde', 78: 'Pale Lagers. Malt Liquor', 79: 'Strong Ales. Old Ale', 80: 'Strong Ales. Imperial Red Ale', 81: 'Wild/Sour Beers. Flanders Red Ale', 82: 'Wild/Sour Beers. Flanders Oud Bruin', 83: 'Strong Ales. English Strong Ale', 84: 'Wheat Beers. Dunkelweizen', 85: 'Pale Lagers. India Pale Lager (IPL)', 86: 'Dark Lagers. Czech Dark Lager', 87: 'Brown Ales. English Dark Mild Ale', 88: 'Porters. Robust Porter', 89: 'Dark Lagers. Rauchbier', 90: 'Specialty Beers. Smoked Beer', 91: 'Specialty Beers. Chile Beer', 92: 'Brown Ales. Belgian Dark Ale', 93: 'Bocks. Weizenbock', 94: 'Wild/Sour Beers. Brett Beer', 95: 'Wild/Sour Beers. Gueuze', 96: 'Specialty Beers. Japanese Rice Lager', 97: 'India Pale Ales. Brut IPA', 98: 'Strong Ales. Wheatwine', 99: 'Pale Lagers. Imperial Pilsner', 100: 'Porters. Smoked Porter', 101: 'Pale Ales. Grisette', 102: 'Pale Lagers. Czech Pale Lager', 103: 'Specialty Beers. Gruit / Ancient Herbed Ale', 104: 'Wheat Beers. Kristallweizen', 105: 'Wild/Sour Beers. Lambic', 106: 'Pale Ales. English Pale Mild Ale', 107: 'Specialty Beers. Kvass', 108: 'Bocks. Eisbock', 109: 'Dark Lagers. Czech Amber Lager', 110: 'Dark Ales. Roggenbier', 111: 'Specialty Beers. Happoshu', 112: 'Wheat Beers. Grodziskie', 113: 'Wheat Beers. American Dark Wheat Beer', 114: 'Specialty Beers. Sahti', 115: 'Wild/Sour Beers. Faro'}
hier_id2group = {0: 'Pale Lagers', 1: 'Pale Ales', 2: 'Strong Ales', 3: 'Stouts', 4: 'India Pale Ales', 5: 'Wild/Sour Beers', 6: 'Specialty Beers', 7: 'Dark Lagers', 8: 'Porters', 9: 'Wheat Beers', 10: 'Brown Ales', 11: 'Bocks', 12: 'Dark Ales'}
hier_id2style = {0: 'American Adjunct Lager', 1: 'Marzen', 2: 'German Pilsner', 3: 'European Pale Lager', 4: 'American Lager', 5: 'Helles', 6: 'Russian Imperial Stout', 7: 'American Imperial Stout', 8: 'Bohemian / Czech Pilsner', 9: 'Sweet / Milk Stout', 10: 'Imperial IPA', 11: 'Tripel', 12: 'American Porter', 13: 'American Stout', 14: 'American IPA', 15: 'English Pale Ale', 16: 'Fruit and Field Beer', 17: 'Oatmeal Stout', 18: 'Hefeweizen', 19: 'Belgian Pale Strong Ale', 20: 'Pumpkin Beer', 21: 'Belgian Dark Strong Ale', 22: 'American Brown Ale', 23: 'American Blonde Ale', 24: 'New England IPA', 25: 'Witbier', 26: 'Saison', 27: 'American Pale Ale', 28: 'Kölsch', 29: 'English Barleywine', 30: 'American Amber / Red Ale', 31: 'American Pale Wheat Beer', 32: 'Imperial Porter', 33: 'Wild Ale', 34: 'American Barleywine', 35: 'Gose', 36: 'Scotch Ale / Wee Heavy', 37: 'Fruited Kettle Sour', 38: 'Light Lager', 39: 'Doppelbock', 40: 'Winter Warmer', 41: 'Berliner Weisse', 42: 'Quadrupel (Quad)', 43: 'Kellerbier / Zwickelbier', 44: 'Belgian Pale Ale', 45: 'Baltic Porter', 46: 'English Brown Ale', 47: 'American Strong Ale', 48: 'English IPA', 49: 'Fruit Lambic', 50: 'Munich Dunkel', 51: 'Low-Alcohol Beer', 52: 'Black IPA', 53: 'English Bitter', 54: 'Schwarzbier', 55: 'English Porter', 56: 'Vienna Lager', 57: 'Bock', 58: 'Extra Special / Strong Bitter (ESB)', 59: 'Herb and Spice Beer', 60: 'Milkshake IPA', 61: 'Maibock', 62: 'Dubbel', 63: 'Irish Red Ale', 64: 'Irish Dry Stout', 65: 'American Amber / Red Lager', 66: 'Altbier', 67: 'Belgian IPA', 68: 'Scottish Ale', 69: 'Rye Beer', 70: 'European / Dortmunder Export Lager', 71: 'Festbier / Wiesnbier', 72: 'Foreign / Export Stout', 73: 'European Strong Lager', 74: 'Belgian Blonde Ale', 75: 'European Dark Lager', 76: 'English Stout', 77: 'Bière de Garde', 78: 'Malt Liquor', 79: 'Old Ale', 80: 'Imperial Red Ale', 81: 'Flanders Red Ale', 82: 'Flanders Oud Bruin', 83: 'English Strong Ale', 84: 'Dunkelweizen', 85: 'India Pale Lager (IPL)', 86: 'Czech Dark Lager', 87: 'English Dark Mild Ale', 88: 'Robust Porter', 89: 'Rauchbier', 90: 'Smoked Beer', 91: 'Chile Beer', 92: 'Belgian Dark Ale', 93: 'Weizenbock', 94: 'Brett Beer', 95: 'Gueuze', 96: 'Japanese Rice Lager', 97: 'Brut IPA', 98: 'Wheatwine', 99: 'Imperial Pilsner', 100: 'Smoked Porter', 101: 'Grisette', 102: 'Czech Pale Lager', 103: 'Gruit / Ancient Herbed Ale', 104: 'Kristallweizen', 105: 'Lambic', 106: 'English Pale Mild Ale', 107: 'Kvass', 108: 'Eisbock', 109: 'Czech Amber Lager', 110: 'Roggenbier', 111: 'Happoshu', 112: 'Grodziskie', 113: 'American Dark Wheat Beer', 114: 'Sahti', 115: 'Faro'}
sep_id2style = {'Pale Lagers': {0: 'American Adjunct Lager', 1: 'German Pilsner', 2: 'European Pale Lager', 3: 'American Lager', 4: 'Helles', 5: 'Bohemian / Czech Pilsner', 6: 'Light Lager', 7: 'Kellerbier / Zwickelbier', 8: 'European / Dortmunder Export Lager', 9: 'Festbier / Wiesnbier', 10: 'European Strong Lager', 11: 'Malt Liquor', 12: 'India Pale Lager (IPL)', 13: 'Imperial Pilsner', 14: 'Czech Pale Lager'}, 'Pale Ales': {0: 'English Pale Ale', 1: 'American Blonde Ale', 2: 'Saison', 3: 'American Pale Ale', 4: 'Kölsch', 5: 'American Amber / Red Ale', 6: 'Belgian Pale Ale', 7: 'English Bitter', 8: 'Extra Special / Strong Bitter (ESB)', 9: 'Irish Red Ale', 10: 'Belgian Blonde Ale', 11: 'Bière de Garde', 12: 'Grisette', 13: 'English Pale Mild Ale'}, 'Strong Ales': {0: 'Tripel', 1: 'Belgian Pale Strong Ale', 2: 'Belgian Dark Strong Ale', 3: 'English Barleywine', 4: 'American Barleywine', 5: 'Scotch Ale / Wee Heavy', 6: 'Quadrupel (Quad)', 7: 'American Strong Ale', 8: 'Old Ale', 9: 'Imperial Red Ale', 10: 'English Strong Ale', 11: 'Wheatwine'}, 'Stouts': {0: 'Russian Imperial Stout', 1: 'American Imperial Stout', 2: 'Sweet / Milk Stout', 3: 'American Stout', 4: 'Oatmeal Stout', 5: 'Irish Dry Stout', 6: 'Foreign / Export Stout', 7: 'English Stout'}, 'India Pale Ales': {0: 'Imperial IPA', 1: 'American IPA', 2: 'New England IPA', 3: 'English IPA', 4: 'Black IPA', 5: 'Milkshake IPA', 6: 'Belgian IPA', 7: 'Brut IPA'}, 'Wild/Sour Beers': {0: 'Wild Ale', 1: 'Gose', 2: 'Fruited Kettle Sour', 3: 'Berliner Weisse', 4: 'Fruit Lambic', 5: 'Flanders Red Ale', 6: 'Flanders Oud Bruin', 7: 'Brett Beer', 8: 'Gueuze', 9: 'Lambic', 10: 'Faro'}, 'Specialty Beers': {0: 'Fruit and Field Beer', 1: 'Pumpkin Beer', 2: 'Low-Alcohol Beer', 3: 'Herb and Spice Beer', 4: 'Rye Beer', 5: 'Smoked Beer', 6: 'Chile Beer', 7: 'Japanese Rice Lager', 8: 'Gruit / Ancient Herbed Ale', 9: 'Kvass', 10: 'Happoshu', 11: 'Sahti'}, 'Dark Lagers': {0: 'Marzen', 1: 'Munich Dunkel', 2: 'Schwarzbier', 3: 'Vienna Lager', 4: 'American Amber / Red Lager', 5: 'European Dark Lager', 6: 'Czech Dark Lager', 7: 'Rauchbier', 8: 'Czech Amber Lager'}, 'Porters': {0: 'American Porter', 1: 'Imperial Porter', 2: 'Baltic Porter', 3: 'English Porter', 4: 'Robust Porter', 5: 'Smoked Porter'}, 'Wheat Beers': {0: 'Hefeweizen', 1: 'Witbier', 2: 'American Pale Wheat Beer', 3: 'Dunkelweizen', 4: 'Kristallweizen', 5: 'Grodziskie', 6: 'American Dark Wheat Beer'}, 'Brown Ales': {0: 'American Brown Ale', 1: 'English Brown Ale', 2: 'Altbier', 3: 'English Dark Mild Ale', 4: 'Belgian Dark Ale'}, 'Bocks': {0: 'Doppelbock', 1: 'Bock', 2: 'Maibock', 3: 'Weizenbock', 4: 'Eisbock'}, 'Dark Ales': {0: 'Winter Warmer', 1: 'Dubbel', 2: 'Scottish Ale', 3: 'Roggenbier'}}

def predict(model, text, labels_fli):
    t = tokenizer(text,
                  padding='max_length',
                  max_length=512,
                  truncation=True,
                  return_tensors="pt")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    model.eval()
    with torch.no_grad():
        mask = t['attention_mask'].to(device)
        input_id = t['input_ids'].squeeze(1).to(device)
        output = model(input_id, mask)
        pred = output.cpu().numpy()
        idx = np.argmax(pred)
        return labels_fli[idx]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search')
def search():
    mode = request.args.get('mode')
    query = request.args.get('query')

    group, style = '', ''

    if mode == 'flat':
        model = torch.load('models/flat/flat.pt', map_location=torch.device('cpu'))
        res = predict(model, query, flat_id2label).split('.')
        group, style = res[0], res[1].strip()

    if mode == 'hier':
        model_1 = torch.load('models/hier/group.pt', map_location=torch.device('cpu'))
        model_2 = torch.load('models/hier/style.pt', map_location=torch.device('cpu'))

        group = predict(model_1, query, hier_id2group)
        style = predict(model_2, 'Group: {}. Review: {}'.format(group, query), hier_id2style)

    if mode == 'hiersep':
        model_1 = torch.load('models/hier/group.pt', map_location=torch.device('cpu'))
        group = predict(model_1, query, hier_id2group)

        calls_list = {'Pale Lagers': 'models/sep/pale_lagers.pt',
                      'Pale Ales': 'models/sep/pale_ales.pt',
                      'Strong Ales': 'models/sep/strong_ales.pt',
                      'Stouts': 'models/sep/stouts.pt',
                      'India Pale Ales': 'models/sep/ipa.pt',
                      'Wild/Sour Beers': 'models/sep/wild-sour_beers.pt',
                      'Specialty Beers': 'models/sep/specialty.pt',
                      'Dark Lagers': 'models/sep/dark_lagers.pt',
                      'Porters': 'models/sep/porters.pt',
                      'Wheat Beers': 'models/sep/wheat.pt',
                      'Brown Ales': 'models/sep/brown_ales.pt',
                      'Bocks': 'models/sep/bocks.pt',
                      'Dark Ales': 'models/sep/dark_ales.pt'}

        model_2 = torch.load(calls_list[group], map_location=torch.device('cpu'))
        style = predict(model_2, query, sep_id2style[group])

    return render_template('index.html', mode=mode, query=query, group=group, style=style)


if __name__ == '__main__':
    app.run()
