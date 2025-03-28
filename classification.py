from data_main import get_dataloader

#use this for training :) augmented dataset with balanced classes, see data_understanding for how to use
dataloader = get_dataloader()
from data_main import get_dataloader

X_data = []
y_data = []

for images, labels in dataloader:
    flattened = images.view(images.size(0), -1).numpy()
    X_data.append(flattened)
    y_data.append(labels.numpy())

X_data = np.concatenate(X_data, axis=0)  
y_data = np.concatenate(y_data, axis=0)  

X_train, X_test, y_train, y_test = train_test_split(
    X_data,
    y_data,
    test_size=0.2,     
    shuffle=True,
    stratify=y_data    
)

clf = DecisionTreeClassifier(random_state=42)  # you can tune hyperparams if desired
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Decision Tree test accuracy: {accuracy * 100:.2f}%")
