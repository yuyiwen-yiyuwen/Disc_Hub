from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

class mlp_DIA_NN:
    def __init__(self,
                 num_nn=5,
                 batch_size=50,
                 hidden_layers=(25, 20, 15, 10, 5),
                 learning_rate=0.003,
                 max_iter=5,
                 debug=True,
                 early_stopping=False,
                 validation_fraction=0.1,
                 n_iter_no_change=5,           
                 tol=1e-4,):           
        """
        Initialize an ensemble of MLP classifiers wrapped in a soft voting classifier.

        Parameters:
        - num_nn: Number of individual MLP models in the ensemble
        - batch_size: Batch size for training each MLP
        - hidden_layers: Tuple specifying hidden layer sizes for each MLP
        - learning_rate: Initial learning rate for training
        - max_iter: Maximum number of iterations per MLP fit
        - debug: If True, run in single-threaded mode (n_jobs=1) for debugging
        - early_stopping: Whether to use early stopping
        - validation_fraction: Proportion of training data to set aside as validation for early stopping
        """
        self.num_nn = num_nn
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.debug = debug
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

        # Create individual MLP classifiers
        self.mlps = [
            MLPClassifier(
                max_iter=self.max_iter,
                shuffle=True,
                random_state=42 + i,
                learning_rate_init=self.learning_rate,
                solver='adam',
                batch_size=self.batch_size,
                activation='relu',
                hidden_layer_sizes=self.hidden_layers,
                early_stopping=self.early_stopping,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                tol=self.tol
            ) for i in range(self.num_nn)
        ]

        # Assign names to each classifier
        self.names = [f'mlp{i}' for i in range(self.num_nn)]

        # Wrap in a VotingClassifier with soft voting
        self.model = VotingClassifier(
            estimators=list(zip(self.names, self.mlps)),
            voting='soft',
            n_jobs=1 if self.debug else self.num_nn
        )


    def fit(self, X, y):
        """
        Fit the ensemble model on training data.

        Parameters:
        - X: Training features
        - y: Training labels
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict class labels for the given input.

        Parameters:
        - X: Input features

        Returns:
        - Predicted class labels
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for the given input.

        Parameters:
        - X: Input features

        Returns:
        - Class probability estimates
        """
        return self.model.predict_proba(X)

    def get_model(self):
        """
        Return the underlying VotingClassifier model.

        Returns:
        - sklearn VotingClassifier object
        """
        return self.model
