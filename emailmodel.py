import pandas as pd 
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader 
from collections import Counter 
import re 
from sklearn.model_selection import train_test_split
# ============================================================ 
# 1. LOAD DATASET 
# ============================================================ 
# CSV should contain: 
# text  -> email message 
# label -> spam / ham 
df = pd.read_csv(r"C:\Users\nmoha\OneDrive\Desktop\mypproject\AI_Models\emails.csv") 
df=df.rename(columns={"spam": "label"})
df = df[["text", "label"]] 
print(df.shape)
# ============================================================ 
# 2. TEXT CLEANING 
# ============================================================ 
def clean_text(text): 
    text = text.lower() 
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text) 
    return text.split() 
df["tokens"] = df["text"].apply(clean_text)
# ============================================================ 
# 3. BUILD VOCABULARY 
# ============================================================ 
counter = Counter() 
for tokens in df["tokens"]: 
    counter.update(tokens) 
# Special tokens 
vocab = { 
    "<PAD>": 0, 
    "<UNK>": 1 
} 
 
# Add words to vocabulary 
for word, count in counter.items(): 
 
    # Ignore very rare words 
    if count >= 2: 
        vocab[word] = len(vocab) 
 
print("Vocabulary size:", len(vocab)) 
 
# ============================================================ 
# 4. CONVERT TEXT → NUMBERS 
# ============================================================ 
 
def encode_text(tokens): 
 
    return [ 
        vocab.get(word, vocab["<UNK>"]) 
        for word in tokens 
    ] 
 
 
df["encoded"] = df["tokens"].apply(encode_text) 
print(len(max(df["encoded"])))

# ============================================================ 
# 5. PAD / TRUNCATE SEQUENCES 
# ============================================================ 
 
MAX_LENGTH = 200 
 
def pad_sequence(sequence): 
 
    # Truncate 
    sequence = sequence[:MAX_LENGTH] 
 
    # Padding 
    if len(sequence) < MAX_LENGTH: 
        sequence += [vocab["<PAD>"]] * ( 
            MAX_LENGTH - len(sequence) 
        ) 
 
    return sequence 
 
 
df["encoded"] = df["encoded"].apply(pad_sequence) 
 
 
# ============================================================ 
# 6. TRAIN / TEST SPLIT 
# ============================================================ 
 
 
X = torch.tensor(df["encoded"].tolist(), dtype=torch.long) 
y = torch.tensor(df["label"].values, dtype=torch.float32) 
 
X_train, X_test, y_train, y_test = train_test_split( 
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y 
) 
print(len(X_train),len(X),len(X_test),len(y_train),len(y),len(y_test))
print(X[0]) 

# ============================================================ 
# 7. PYTORCH DATASET 
# ============================================================ 
 
class SpamDataset(Dataset): 
 
    def __init__(self, X, y): 
        self.X = X 
        self.y = y 
 
    def __len__(self): 
        return len(self.X) 
 
    def __getitem__(self, index): 
        return self.X[index], self.y[index] 
 
 
train_dataset = SpamDataset(X_train, y_train) 
test_dataset = SpamDataset(X_test, y_test)
 
 
train_loader = DataLoader( 
    train_dataset, 
    batch_size=1, 
    shuffle=True 
) 
 
test_loader = DataLoader( 
    test_dataset, 
    batch_size=1 
) 
 
print(len(train_loader))
print(test_loader)  
 
# ============================================================ 
# 8. BUILD LSTM MODEL 
# ============================================================ 
 
class SpamLSTM(nn.Module): 
 
    def __init__( 
        self, 
        vocab_size, 
        embedding_dim=128, 
        hidden_dim=128 
    ): 
 
        super().__init__() 
 
        # Word embeddings 
        self.embedding = nn.Embedding( 
            vocab_size, 
            embedding_dim, 
            padding_idx=0 
        ) 
 
        # LSTM 
        self.lstm = nn.LSTM( 
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            batch_first=True 
        ) 
 
        # Fully connected layer 
        self.fc = nn.Linear( 
            hidden_dim, 
            1 
        ) 
 
    def forward(self, x): 
 
        # [batch, sequence_length] 
        x = self.embedding(x) 
 
        # [batch, sequence_length, embedding_dim] 
        output, (hidden, cell) = self.lstm(x) 
 
        # Last hidden state 
        x = hidden[-1] 
 
        # Classification 
        x = self.fc(x) 
 
        return x.squeeze(1) 
 
 
# ============================================================ 
# 9. CREATE MODEL 
# ============================================================ 
 
device = torch.device( 
    "cuda" if torch.cuda.is_available() else "cpu" 
) 
 
model = SpamLSTM( 
    vocab_size=len(vocab) 
).to(device) 
print(device) 
print(model) 
 
 
# ============================================================ 
# 10. LOSS + OPTIMIZER 
# ============================================================ 
 
criterion = nn.BCEWithLogitsLoss() 
 
optimizer = torch.optim.Adam( 
    model.parameters(), 
    lr=0.001 
) 
 
 
# ============================================================ 
# 11. TRAIN MODEL 
# ============================================================ 
 
EPOCHS = 10 
 
for epoch in range(EPOCHS): 
 
    model.train() 
 
    total_loss = 0 
 
    for emails, labels in train_loader: 
 
        emails = emails.to(device) 
        labels = labels.to(device) 
 
        # Forward pass 
        outputs = model(emails) 
 
        # Loss 
        loss = criterion(outputs, labels) 
 
        # Clear gradients 
        optimizer.zero_grad() 
 
        # Backpropagation 
        loss.backward() 
 
        # Update weights 
        optimizer.step() 
 
        total_loss += loss.item() 
 
    print( 
        f"Epoch [{epoch+1}/{EPOCHS}] " 
        f"Loss: {total_loss / len(train_loader):.4f}" 
    ) 
 
 
# ============================================================ 
# 12. TEST MODEL 
# ============================================================ 
 
model.eval() 
 
correct = 0 
total = 0 
 
with torch.no_grad(): 
 
    for emails, labels in test_loader: 
 
        emails = emails.to(device) 
        labels = labels.to(device) 
 
        outputs = model(emails) 
 
        probabilities = torch.sigmoid(outputs) 
 
        predictions = (probabilities >= 0.5).float() 
 
        correct += ( 
            predictions == labels 
        ).sum().item() 
 
        total += labels.size(0) 
 
 
accuracy = correct / total 

 
print(f"Test Accuracy: {accuracy * 100:.2f}%") 

torch.save({"model_state_dist":model.state_dict(),
            "vocab":vocab,
            "vocab_size":len(vocab),
            "embedding_dim":128,
            "hidden_dim":128,
            "max_length":MAX_LENGTH,
            },"spam_model.pth")
print("model saved")  
# ============================================================ 
# 13. PREDICT NEW EMAIL 
# ============================================================ 
 
def predict_email(email): 
 
    model.eval() 
 
    # Clean 
    tokens = clean_text(email) 
 
    # Encode 
    encoded = encode_text(tokens) 
 
    # Pad 
    encoded = pad_sequence(encoded) 
 
    # Tensor 
    tensor = torch.tensor( 
        [encoded], 
        dtype=torch.long 
    ).to(device) 
 
    with torch.no_grad(): 
 
        output = model(tensor)
        print(output) 
 
        probability = torch.sigmoid(output).item() 
 
    if probability >= 0.5: 
        result = "SPAM " 
    else: 
        result = "NOT SPAM " 
 
    print("Email:", email) 
    print("Spam Probability:", round(probability, 4)) 
    print("Prediction:", result) 
 
 
# ============================================================ 
# 14. TEST 
# ============================================================ 
 
predict_email( 
    "Subject: the stock trading gunslinger  fanny is merrill but muzo not colza attainder and penultimate like esmark perspicuous ramble is segovia not group try slung kansas tanzania yes chameleon or continuant clothesman no  libretto is chesapeake but tight not waterway herald and hawthorn like chisel morristown superior is deoxyribonucleic not clockwork try hall incredible mcdougall yes hepburn or einsteinian earmark no  sapling is boar but duane not plain palfrey and inflexible like huzzah pepperoni bedtime is nameable not attire try edt chronography optima yes pirogue or diffusion albeit no " 
) 
 
predict_email( 
    "Subject: re : tony hamilton  tony hamilton reports to mike roberts in the houston office .  vince kaminski will answer these questions for you at any  given time .  tony ' s start date in london will be april 9 th  thanks  kevin moore  desleigh langfield  04 / 04 / 2001 08 : 29 am  to : kevin g moore / hou / ect @ ect  cc :  subject : tony hamilton  kevin  tani nath who heads up the structuring and research teams rang this morning  to get more information on tony .  can you tell me who he reports to in houston and whether that person will  continue to manage him remotely ?  what cost centre he should be charged to ? whose headcount he appears on , and  for what group he will be providing services for , tani was unsure on all of  the above and wants clarification .  also can you confirm his start date in london  thanks  desleigh  - - - - - - - - - - - - - - - - - - - - - - forwarded by desleigh langfield / lon / ect on 04 / 04 / 2001  14 : 27 - - - - - - - - - - - - - - - - - - - - - - - - - - -  desleigh langfield  06 / 03 / 2001 13 : 40  to : steven leppard / lon / ect @ ect  cc :  subject : tony hamilton  steve  all sorted and we will check later in the week with it that all is okay .  can you please tell your assistant to organise a desk for tony , no rush  obviously  thanks  desleigh  - - - - - - - - - - - - - - - - - - - - - - forwarded by desleigh langfield / lon / ect on 06 / 03 / 2001  13 : 38 - - - - - - - - - - - - - - - - - - - - - - - - - - -  desleigh langfield  06 / 03 / 2001 13 : 38  to : european resolution center / lon / ect @ ect  cc : kevin g moore / hou / ect @ ect , steven leppard / lon / ect @ ect , anna  seymour / lon / ect @ ect  subject : tony hamilton  hi  tony is a uk employee who starts next monday 12 th march 2001 , we have done a  quick start in nest today for him .  tony will be working his first month in the houston office , however we still  need to set up a log on and notes account here .  can you please send the log on and password for both accounts to kevin as he  will be meeting with tony on monday morning in the houston office .  tony will not have a desk arranged for him until he comes back in  approximately a month so no actual pc is necessary until then .  one question - with a uk log on and notes account , will tony be able to  access these from houston ? if it ' s complicated can you please let kevin know  how to do this .  any problems let me know  thanks  desleigh" 
)
#print(vocab) 
#print(model.state_dict())
#print(len(vocab))
#print(MAX_LENGTH)
