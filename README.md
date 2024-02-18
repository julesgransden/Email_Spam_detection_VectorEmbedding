# Email_Spam_detection_VectorEmbedding
In this project I look into the accuracy of a spam detection algorithm which utilizes openAI's vector embedding API to represent our data. 


## Data
source: https://www.kaggle.com/datasets/mfaisalqureshi/spam-email
initial raw data:

<img width="372" alt="Screen Shot 2024-02-18 at 2 54 41 PM" src="https://github.com/julesgransden/Email_Spam_detection_VectorEmbedding/assets/78057184/e81ed54a-f1ed-4747-aaa5-a71d74aa915b">
After vectorization:
<img width="946" alt="Screen Shot 2024-02-18 at 2 55 56 PM" src="https://github.com/julesgransden/Email_Spam_detection_VectorEmbedding/assets/78057184/67c424a3-6100-46b2-b836-10fed96b2ebc">
As we can observe, the embedding API turned each email into a vector with 1536 features.

## Comparing accuracy
- Decision tree: 96.19%
  <img width="977" alt="Screen Shot 2024-02-18 at 3 16 08 PM" src="https://github.com/julesgransden/Email_Spam_detection_VectorEmbedding/assets/78057184/7d8923d7-22be-45ce-b185-1d57f50a4669">

- Random Forrest: 98.35%
- Logistic Regression: 98.92%
- Artificial Neural Network: 99.569%
- SVM-Support vector machines: 99.7%

## Run File
- Install required Libraries
- Clone repository 
- and run File
