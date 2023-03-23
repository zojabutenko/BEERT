## 🍺 BEERT - Beer Advisor With BERT

### Welcome to Beert, a neural network(s) that helps you choose a beer for an evening with just one query. The network can recommend  both a group and a style of the beer. 


The data that the networks were trained with are reviews from the [BeerAdvocate](https://www.beeradvocate.com/) website.

[Models](https://drive.google.com/drive/folders/1Uvz88ys4n1u0cPb2gL9ieMg1m59PpYaE)

The following approaches have been tested:

### [**1. Flat Approach**](https://github.com/zojabutenko/BEERT/tree/main/notebooks/flat)

In this approach, we train a single neural network that returns a group and style of beer at once. It takes a review as input, and returns a Group+Style fusion as output.

![enter image description here](https://i.ibb.co/F4dFKs1/flat.png)

### [2. Hierarchical Approach](https://github.com/zojabutenko/BEERT/tree/main/notebooks/hierarchical)

In this approach, we train two neural networks. The first one is trained on reviews and returns a group of the beer. The second one is trained on the fusions of Group and Reviews and returns a style.

![enter image description here](https://i.ibb.co/80PNHGB/hier.png)

### [3. **Hierarchical-Separate Approach**](https://github.com/zojabutenko/BEERT/tree/main/notebooks/hierarchical-separate)

In this approach, we use the first network from Hierarchical approach that predicts the Group, and train separate neural networks for each group (13 in total) that return a style.

![enter image description here](https://i.ibb.co/7jPHd54/hier-sep.png)

The models are available for download [here](https://drive.google.com/file/d/1siVzANkbX6Mi2NGZkehBhC4vV8PeyPZu/view?usp=share_link).

## User Interface for the  models

To be able to conveniently use the models, we have developed a user interface that looks as follows:
![enter image description here](https://i.ibb.co/StNN49D/web.png)
