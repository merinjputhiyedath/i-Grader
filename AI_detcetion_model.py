#!/usr/bin/env python
# coding: utf-8

# # Merin Joseph
# ## CPSC 597-project: i-Grader ( Faculty Assist System: AI Detection)
# ### CSUF - CS Grad
# ### CWID: 885869974

# In[30]:


import pandas as pd
from sklearn.metrics import accuracy_score
import pickle
import torch
from transformers import pipeline
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, AdamW


# In[31]:


pipe = pipeline("text-classification", model="roberta-base")


# In[32]:


tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForMaskedLM.from_pretrained("roberta-base")


# In[33]:


data_test = '''Laughter is a universal language that transcends cultural, linguistic, and social barriers. It is a powerful and instinctive human expression that brings joy, connection, and numerous health benefits. The importance of laughter in our lives cannot be overstated, as it plays a pivotal role in enhancing our physical, mental, and emotional well-being.On a physiological level, laughter triggers a cascade of positive effects within the body. It promotes the release of endorphins, often referred to as "feel-good" hormones, which contribute to a sense of happiness and euphoria. The act of laughing also stimulates the cardiovascular system, increasing blood flow and improving heart health. Additionally, laughter has been shown to boost the immune system, providing a natural defense against illness and stress.'''


# In[34]:


data_test_human='''Laughter for me is very important as it is what hold my family together. we have our qaulity family time everyday at night where we share what happened in our day and so on. This makes me so happy. I love my family.'''


# In[35]:


file_path = r"C:\Users\ponny\Desktop\final_proj\wiki-labeled.csv"  
df = pd.read_csv(file_path)


# In[39]:


texts = df['text'].tolist()[:30000]
labels = df['label'].tolist()[:30000]


# In[43]:


tokenizer = AutoTokenizer.from_pretrained("andreas122001/roberta-wiki-detector")


# In[44]:


tokenized_data = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
# outputs = model(**inputs)


# In[50]:


train_inputs, val_inputs, train_labels, val_labels = train_test_split(tokenized_data['input_ids'], labels, test_size=0.2, random_state=42)


# In[51]:


optimizer = AdamW(model.parameters(), lr=5e-5)


# In[57]:


print(train_inputs[1])


# In[66]:


# Fine-tune the model
model.train()
for epoch in range(3):  
    optimizer.zero_grad()
    
    outputs = model(train_inputs, labels=train_labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()


# In[ ]:


# Save the tokenizer and model using pickle

pickle.dump(tokenizer, open(r"C:\Users\ponny\Desktop\final_proj\webapp\aitokenizer.pkl", 'wb'))
pickle.dump(model,open(r"C:\Users\ponny\Desktop\final_proj\webapp\aimodel.pkl", 'wb'))

# Load the tokenizer and model back using pickle
loaded_tokenizer = pickle.load(open(r"C:\Users\ponny\Desktop\final_proj\webapp\aitokenizer.pkl", 'rb'))
loaded_model = pickle.load(open(r"C:\Users\ponny\Desktop\final_proj\webapp\aimodel.pkl", 'rb'))


# In[28]:


if predicted_class == 1:
    print("The sample text is AI-generated or AI-related.")
else:
    print("The sample text is human-authored or not AI-related.")


# In[ ]:


df_subset = df.head(2000)


# In[ ]:


texts = df_subset['text'].tolist()


# In[ ]:


true_labels = df_subset['label'].tolist()


# In[ ]:


predicted_labels = []


# In[ ]:


print(len(texts))


# In[22]:


for i in range(len(texts)):
    text=texts[i]
    true_label=true_labels[i]
    # Tokenize the text
    inputs = loaded_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Make predictions
    outputs = loaded_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Append the predicted label to the list
    predicted_labels.append(predicted_class)
    print(f"Processing row {i + 1}: Predicted class - {predicted_class}, True label - {true_label}")


# In[28]:


accuracy = accuracy_score(true_labels, predicted_labels)



# In[29]:


print(f'Accuracy on the first 2000 rows: {accuracy*100}')


# In[ ]:





# In[ ]:




