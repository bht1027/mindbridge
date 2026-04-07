import json
from pathlib import Path


SRC = Path("/Users/shaoyuzhe/Downloads/ADL_HW_3A_MLP_Text_Classification.ipynb")
DST = Path("/Users/shaoyuzhe/Documents/New project/ADL_HW_3A_MLP_Text_Classification_completed.ipynb")


def set_source(cell, source):
    if isinstance(source, str):
        source = source.splitlines(keepends=True)
    cell["source"] = source


nb = json.loads(SRC.read_text())
cells = nb["cells"]

set_source(
    cells[6],
    """from torch._functorch.vmap import lazy_load_decompositions
# Short Question

torch.manual_seed(SEED)
# Define y to be a target of dimension (1, 3) without a gradient
y = torch.randn(1, 3)

# Define theta to be a random tensor of dimension (1, 3) which requires a gradient; we want theta to converge to y
theta = torch.randn(1, 3, requires_grad=True)


# Define an SGD optimizer with learning rate 0.01 which acts on theta
optimizer = torch.optim.SGD([theta], lr=0.01)

# Fil in the code below using the optimizer above to get theta to converge to y
for epoch in range(100):
  # Zero out the gradients of l with respect to theta
  optimizer.zero_grad()

  # Define a loss manually which is ||theta-x||_{2}^{2}, the L2 loss across all components
  loss = ((theta - y) ** 2).sum()

  print('Epoch:{} Loss: {}'.format(epoch, loss))

  # Get teh gradients of l with respect to theta
  loss.backward()

  # Update theta
  optimizer.step()

# These should look very similar
print(y)
print(theta)
with torch.no_grad():
  # Check the y and theta have converged to almost the same thing
  loss = ((theta - y) ** 2).sum()
  assert (loss.item() - 0.0)**2 <= 0.001
""",
)

set_source(
    cells[7],
    """# Suppose we forget optimizer.zero_grad()
# Given an example of what this does and why we WOULD want to do this
# Hint: if you are doing batch gradient descent and call optimizer.zero_grad() every 3 batches, what is the gradient represent?

\"\"\"
Gradients accumulate by default in PyTorch. If you skip optimizer.zero_grad(),
the next backward() call adds its gradient to the existing one instead of
replacing it. For example, if you zero every 3 batches, then the stored
gradient is the sum of the gradients from those 3 batches, which is equivalent
to using a larger effective batch before taking one optimizer step.
\"\"\"
""",
)

set_source(
    cells[14],
    """# A basic tokenizer by using get_tokenizer; pass "basic_english"
basic_english_tokenizer = get_tokenizer("basic_english")
""",
)

set_source(
    cells[18],
    """# Loop through all the (label, text) data and yield a tokenized version of text
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield TOKENIZER(text)
""",
)

set_source(
    cells[29],
    """from torchtext.vocab.vocab_factory import Vocab
# Utility to transform text into a list of ints
# This shoould go "a b c" -> ["a", "b", "c"] -> [1, 2, 3], for example
def text_pipeline(x):
    # Apply tokenizer to x
    x = TOKENIZER(x)

    # Return the Vocab at those tokens
    return VOCAB(x)

# Return a 0 starting version of x
# If x = "1" this should return 0
# If x = "3" this should return 2, Etc.
def label_pipeline(x):
    return int(x) - 1
""",
)

set_source(
    cells[31],
    """# For a batch of data that might not be a tensor, return the batch in ternsor version
# batch is a length B lsit of tuples where each element is (label, text)
# label is a raw string like "1" here; text is a sentence like "this is about soccer"
def collate_batch(batch):
    label_list, text_list = [], []
    for (label, text) in batch:
        # Get the label from {1, 2, 3, 4} to {0, 1, 2, 3} and append it to label list
        label_list.append(label_pipeline(label))

        # Return a list of ints
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)

    # Make label_list into a tensor of dtype=torch.int64
    label_list = torch.tensor(label_list, dtype=torch.int64)

    # Pad the sequence
    # For Exmaple: if we had 2 elements and [[1, 2], [1,2,3,4]] in the text_list then we want
    # to have [[1, 2, 0, 0], [1, 2, 3, 4]] in text_list and text_list is a tensor
    # Look up pad_sequence and make sure you specify batch_first=True and specify the padding_value=0
    text_list = pad_sequence(text_list, batch_first=True, padding_value=PADDING_VALUE)

    # Return the data and put it on a GPU or CPU, as needed
    return label_list.to(DEVICE), text_list.to(DEVICE)
""",
)

set_source(
    cells[34],
    """# Get an iterator for the AG_NEWS dataset and get the train version
train_iter = DATASETS[DATASET](root=DATA_DIR, split="train")

# Use the above to get the number of class elements
num_class = len(set(label for label, _ in train_iter))
# What are the classes?
print(f"The number of classes is {num_class} ...")
""",
)

set_source(
    cells[37],
    """# A very naive model used to classify text
class OneHotTextClassificationModel(nn.Module):
    def __init__(self, vocab_size, num_class):
        super(OneHotTextClassificationModel, self).__init__()
        self.vocab_size = vocab_size
        self.num_class = num_class

        # Have this layer take in data of dimension vocab_size and return data of dimension 100
        # Don't use a bias
        self.fc1 = nn.Linear(vocab_size, 100, bias=False)

        # We will not use this, but see below as we want to mimic this layer using one_hot and fc1
        self.e = nn.Embedding(vocab_size, 100)

        # Have this layer take in 100 and return data of dimension num_class
        # Don't use a bias
        self.fc2 = nn.Linear(100, num_class, bias=False)
        self.init_weights()

        # See forward below; we do not use this but you can use this if you want to to check
        self.use_embedding_layer = False

    def init_weights(self):
        # Initialize the weights of fc1 to the same exact data as what self.e has
        # You need to access the data within these layers
        # Initialize the bias to zero
        # Hint: look at self.e.weight.data and similarly for fc
        # Make sure you have the dimensions line up right
        self.fc1.weight.data = self.e.weight.data.T.clone()

        # Unitialize fc2 to uniform between -0.5 and 0.5
        # Hint: "uniform_"
        initrange = 0.5
        self.fc2.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        B, K = x.shape
        # x is of dimension (B, K), where K is the maximum number of tokens in an element of the batch
        # Note: We will make this faster later on by using the nn.Embedding layer

        # We will not use nn.Embedding, but the code below, a combination of F.one_hot and fc1, should be the SAME effect as the else clause
        if not self.use_embedding_layer:
          # Transform x to a tensor where each element is one-hot encoded
          x = F.one_hot(x, num_classes=self.vocab_size).float()
          assert(x.shape == (B, K, self.vocab_size))

          # Pass x through fc1 to get the row in fc1 correspondng to the row x is
          x = self.fc1(x)
          assert(x.shape == (B, K, 100))
        else:
          # Note: the above two steps should be the same as doing the command below
          x = self.e(x)
          assert(x.shape == (B, K, 100))

        # Take the mean of the embedings for all words in each sentence
        x = x.mean(dim=1)
        assert(x.shape == (B, 100))

        # Apply ReLU to x
        x = F.relu(x)
        assert(x.shape == (B, 100))

        # Pass through fc2
        x = self.fc2(x)
        assert(x.shape == (B, self.num_class))

        # Return the Logits
        return x
""",
)

set_source(
    cells[40],
    """# Map the data to the right format
train_iter, test_iter = DATASETS[DATASET]()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)

# Split data into train and validation
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

# Set up different DataLoaders
# Make sure you pass collate_fn as the function you wrote above
train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
""",
)

set_source(
    cells[43],
    """def train(dataloader, model, optimizer, criterion, epoch):
    # Put the model in train mode; this does not matter right now
    model.train()
    total_acc, total_count = 0, 0
    total_loss = 0.0
    log_interval = 200

    for idx, (label, text) in enumerate(dataloader):
        # Zero out the gradients
        optimizer.zero_grad()

        # Get the predictions
        predicted_label = model(text)

        # Get the loss.
        loss = loss_fn(input=predicted_label, target=label)

        # The loss is computed by taking a mean, get the sum of the terms on the numerator
        with torch.no_grad():
          total_loss += loss.item() * label.size(0)

        # Do back propagation
        loss.backward()

        # Clip the gradients to have max norm 0.1
        # Look up torch.nn.utils.clip_grad_norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        # Do an optimization step.
        optimizer.step()

        # Get the accuracy
        # predicted_label is (B, num_class) so take the argmax over the right dimension to get the actual label
        # Make sure you do .item() on whaht you get so that you update the accuracy
        total_acc += (predicted_label.argmax(dim=1) == label).sum().item()

        # Update the total number of items
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f} "
                "| loss {:8.3f}".format(
                    epoch, idx,
                    len(dataloader),
                    total_acc / total_count,
                    total_loss / total_count
                    )
            )
            total_acc, total_count, total_loss = 0, 0, 0.0
""",
)

set_source(
    cells[44],
    """def evaluate(dataloader, model):
    # Put the model in eval model; this does not matter right now
    model.eval()
    accuracy_fn = lambda logits, target: (logits.argmax(dim=1) == target).float().mean()

    # Get the accuracy
    total_acc = 0.0
    total_count = 0.0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            # Get the predictions
            predicted_label = model(text)
            # Get the number of samples we have, the denominator of accuracy
            total_count += label.size(0)

            # Get the total number of times we have the correct predictions, use accuracy_fn
            total_acc += (predicted_label.argmax(dim=1) == label).sum().item()

            # Use accuracy_fn from torchmetrics to check that the total number of correct predictions is the same as if you use argmax on predicted_label
            # I.e. I want you to use torchmetrics to compute this AND use the same metod as in train above
            # Remember to use .item() on the tensor you get and also rememeber number_or_samples * accuracy = total_times_we_have_equality (the numerator of accuracy)
            assert (
                accuracy_fn(predicted_label, label).item() * label.size(0) == (predicted_label.argmax(dim=1) == label).sum().item()
            )

    accuracy = total_acc / total_count
    return accuracy
""",
)

set_source(
    cells[46],
    """# Set up the loss function
# Note that this should be a multiclass classification problem and you take in logits
loss_fn = nn.CrossEntropyLoss().to(DEVICE)

# Instantiate the model
# Pass in the number of elements in VOCAB and num_class
model = OneHotTextClassificationModel(len(VOCAB), num_class).to(DEVICE)

# Instantiate the SGD optimizer with parameters LR
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
""",
)

set_source(
    cells[48],
    """# Optional: verify the one-hot path and embedding path produce the same logits
# after syncing fc1 with the embedding matrix.
with torch.no_grad():
    model.fc1.weight.data.copy_(model.e.weight.data.T)
    sample_labels, sample_text = next(iter(test_dataloader))
    model.use_embedding_layer = True
    embedding_logits = model(sample_text)
    model.use_embedding_layer = False
    one_hot_logits = model(sample_text)
    print('max |embedding - one_hot|:', (embedding_logits - one_hot_logits).abs().max().item())
""",
)

DST.write_text(json.dumps(nb, indent=1))
print(DST)
