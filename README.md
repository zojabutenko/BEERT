## 🍺 BEERT - Beer Advisor With BERT

### Welcome to Beert, a neural network(s) that helps you choose a beer for an evening with just one query. The network can recommend  both a group and a style of the beer. 


The data that the networks were trained with are reviews from the [BeerAdvocate](https://www.beeradvocate.com/) website.

The following approaches have been tested:

### [**1. Flat Approach**](https://github.com/zojabutenko/BEERT/tree/main/notebooks/flat)

In this approach, we train a single neural network that returns a group and style of beer at once. It takes a review as input, and returns a Group+Style fusion as output.

![enter image description here](https://i.ibb.co/F4dFKs1/flat.png)

The model is available for download [here](https://drive.google.com/file/d/1-6PPQumcDLr9wE5PGQ55g_2Hxdn6WZ1e/view?usp=share_link).

### [2. Hierarchical Approach](https://github.com/zojabutenko/BEERT/tree/main/notebooks/hierarchical)

In this approach, we train two neural networks. The first one is trained on reviews and returns a group of the beer. The second one is trained on the fusions of Group and Reviews and returns a style.

![enter image description here](https://i.ibb.co/80PNHGB/hier.png)

The first model is available [here](https://drive.google.com/file/d/1Snw6Wq4QG6K2wotjiQGnML5yxnBCNpD4/view?usp=share_link) and the second classifier is [here](https://drive.google.com/file/d/1aeMgECcim2KzRJV9R3pmJejLDNEqU58O/view?usp=share_link).

### [3. **Hierarchical-Separate Approach**](https://github.com/zojabutenko/BEERT/tree/main/notebooks/hierarchical-separate)

In this approach, we use the first network from Hierarchical approach that predicts the Group, and train separate neural networks for each group (13 in total) that return a style.

![enter image description here](https://i.ibb.co/7jPHd54/hier-sep.png)

The 13 style-separate models are available for download [here](https://drive.google.com/file/d/12J5ke9iyChuFMHWps-RJsJyJeXyU6wg1/view?usp=share_link).

## User Interface for the  models

To be able to conveniently use the models, we have developed a user interface that looks as follows:
![enter image description here](https://i.ibb.co/StNN49D/web.png)
