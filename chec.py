import torch 
import torch.nn as nn
import re
class SpamLSTM(nn.Module): 
    def __init__( 
        self, 
        vocab_size, 
        embedding_dim=128, 
        hidden_dim=128 
    ):
        super().__init__() 
        self.embedding = nn.Embedding( 
            vocab_size, 
            embedding_dim, 
            padding_idx=0 
        ) 
        self.lstm = nn.LSTM( 
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            batch_first=True 
        ) 
        self.fc = nn.Linear( 
            hidden_dim, 
            1 
        ) 
    def forward(self, x):
        x = self.embedding(x) 
        output, (hidden, cell) = self.lstm(x) 
        x = hidden[-1] 
        x = self.fc(x) 
        return x.squeeze(1)   
device = torch.device( 
    "cuda" if torch.cuda.is_available() else "cpu" 
)  
checkpoint=torch.load("spam_model.pth",map_location=device)
vocab= checkpoint["vocab"]
vocab_size=checkpoint["vocab_size"]
embedding_dim=checkpoint["embedding_dim"]
hidden_dim=checkpoint["hidden_dim"]
MAX_LENGTH=checkpoint["max_length"] 
model = SpamLSTM( 
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim, 
).to(device) 
model.load_state_dict(checkpoint["model_state_dist"])
def clean_text(text): 
    text = text.lower() 
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text) 
    return text.split() 
def encode_text(tokens):
    return [ 
        vocab.get(word, vocab["<UNK>"]) 
        for word in tokens 
    ] 
def pad_sequence(sequence): 
    # Truncate 
    sequence = sequence[:MAX_LENGTH] 
    # Padding 
    if len(sequence) < MAX_LENGTH: 
        sequence += [vocab["<PAD>"]] * ( 
            MAX_LENGTH - len(sequence) 
        ) 
    return sequence
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
        probability = torch.sigmoid(output).item() 
        return (round(probability,4))
aee="subject : Action required:Update payroll Verification.\n\nAs part of routine payroll system update,all employees are required to verify their banking information before the next processing cycle.\n\nYour current payroll details will remain unchanged unless you choose to update them.Please review the information through the employee portal and confirm that your account details are correct.\n\nFor security reasons,donot reply this email with your password,PIN,or credentials.\n\n If you have already completed the verification no further action is required.\n\nRegards,\nPayroll Administration"  
a=predict_email(aee)  
print(a)