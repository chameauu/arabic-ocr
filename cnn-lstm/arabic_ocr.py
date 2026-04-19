import os
import numpy as np
import tensorflow as tf
import Config as config


class ArabicOCR(tf.keras.Model):

    def __init__(self, decoder_type=config.DecoderType.BestPath, must_restore=False, dump=False):
        """Initialize model: add CNN, RNN and CTC layers"""
        super(ArabicOCR, self).__init__()
        
        self.dump = dump
        with open(config.fnCharList, encoding="utf-8") as f:
            self.char_list = f.read()
        self.decoder_type = decoder_type
        self.must_restore = must_restore
        self.snap_id = 0
        self.batches_trained = 0
        
        # Build the model architecture
        self._build_cnn()
        self._build_rnn()
        
        # Setup optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        
        # Load weights if needed
        if must_restore:
            self._restore_model()
    
    def _build_cnn(self):
        """Create CNN layers (5-layer architecture)"""
        self.cnn_layers = [
            # Layer 1: 5x5 conv, 32 filters
            tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
            
            # Layer 2: 5x5 conv, 64 filters
            tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
            
            # Layer 3: 3x3 conv, 128 filters
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((1, 2), strides=(1, 2)),
            
            # Layer 4: 3x3 conv, 128 filters
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((1, 2), strides=(1, 2)),
            
            # Layer 5: 3x3 conv, 256 filters
            tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((1, 2), strides=(1, 2)),
        ]
    
    def _build_rnn(self):
        """Create RNN layers"""
        num_hidden = 256
        
        # Bidirectional LSTM layers
        self.rnn_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(num_hidden, return_sequences=True),
            merge_mode='concat'
        )
        
        # Output projection layer
        self.output_layer = tf.keras.layers.Dense(len(self.char_list) + 1)
    
    def call(self, inputs, training=False):
        """Forward pass through the network"""
        # Add channel dimension if needed
        if len(inputs.shape) == 3:
            x = tf.expand_dims(inputs, axis=-1)
        else:
            x = inputs
        
        # CNN layers
        for layer in self.cnn_layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Squeeze height dimension for RNN
        x = tf.squeeze(x, axis=2)
        
        # RNN layers
        x = self.rnn_layer(x, training=training)
        
        # Output projection
        x = self.output_layer(x)
        
        return x
    
    def compute_ctc_loss(self, y_true_sparse, y_pred, input_length, label_length):
        """Compute CTC loss"""
        return tf.nn.ctc_loss(
            labels=y_true_sparse,
            logits=y_pred,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=len(self.char_list)
        )
    
    def decode_predictions(self, y_pred, input_length):
        """Decode CTC predictions to text"""
        if self.decoder_type == config.DecoderType.BestPath:
            decoded, _ = tf.nn.ctc_greedy_decoder(
                tf.transpose(y_pred, [1, 0, 2]),
                input_length
            )
        elif self.decoder_type == config.DecoderType.BeamSearch:
            decoded, _ = tf.nn.ctc_beam_search_decoder(
                tf.transpose(y_pred, [1, 0, 2]),
                input_length,
                beam_width=50
            )
        else:
            raise NotImplementedError("WordBeamSearch not implemented in TF2 version")
        
        return decoded[0]
    
    def to_sparse(self, texts):
        """Convert ground truth texts to sparse tensor format"""
        indices = []
        values = []
        
        # Handle empty texts
        if not texts or all(len(t) == 0 for t in texts):
            return tf.SparseTensor(
                indices=tf.zeros((0, 2), dtype=tf.int64),
                values=tf.zeros((0,), dtype=tf.int32),
                dense_shape=[len(texts) if texts else 1, 1]
            )
        
        max_len = 0
        for batch_idx, text in enumerate(texts):
            if len(text) > 0:
                char_indices = [self.char_list.index(c) for c in text if c in self.char_list]
                max_len = max(max_len, len(char_indices))
                for pos, char_idx in enumerate(char_indices):
                    indices.append([batch_idx, pos])
                    values.append(char_idx)
        
        # Ensure we have at least one column
        if max_len == 0:
            max_len = 1
        
        return tf.SparseTensor(
            indices=indices if indices else [[0, 0]],
            values=values if values else [0],
            dense_shape=[len(texts), max_len]
        )
    
    def decoder_output_to_text(self, sparse_tensor, batch_size):
        """Convert sparse tensor output to text strings"""
        texts = [[] for _ in range(batch_size)]
        
        for idx in range(len(sparse_tensor.indices)):
            batch_idx = sparse_tensor.indices[idx][0]
            char_idx = sparse_tensor.values[idx]
            texts[batch_idx].append(self.char_list[char_idx])
        
        return [''.join(text) for text in texts]
    
    def train_step(self, images, gt_texts_sparse, seq_lengths):
        """Single training step"""
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self(images, training=True)
            
            # Compute loss
            loss = tf.reduce_mean(
                tf.nn.ctc_loss(
                    labels=gt_texts_sparse,
                    logits=predictions,
                    label_length=None,
                    logit_length=seq_lengths,
                    logits_time_major=False,
                    blank_index=len(self.char_list)
                )
            )
        
        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss
    
    def train_batch(self, batch):
        """Train on a batch of data"""
        num_batch_elements = len(batch.imgs)
        sparse = self.to_sparse(batch.gtTexts)
        
        # Adjust learning rate based on training progress
        if self.batches_trained < 10:
            lr = 0.01
        elif self.batches_trained < 10000:
            lr = 0.001
        else:
            lr = 0.0001
        
        self.optimizer.learning_rate.assign(lr)
        
        # Convert to tensors
        images = tf.constant(batch.imgs, dtype=tf.float32)
        seq_lengths = tf.constant([config.MAX_TEXT_LENGTH] * num_batch_elements, dtype=tf.int32)
        
        # Train step
        loss = self.train_step(images, sparse, seq_lengths)
        
        self.batches_trained += 1
        return loss.numpy()
    
    def infer_batch(self, batch, calc_probability=False, probability_of_gt=False):
        """Run inference on a batch"""
        num_batch_elements = len(batch.imgs)
        
        # Convert to tensor
        images = tf.constant(batch.imgs, dtype=tf.float32)
        seq_lengths = tf.constant([config.MAX_TEXT_LENGTH] * num_batch_elements, dtype=tf.int32)
        
        # Forward pass
        predictions = self(images, training=False)
        
        # Decode
        decoded = self.decode_predictions(predictions, seq_lengths)
        texts = self.decoder_output_to_text(decoded, num_batch_elements)
        
        # Calculate probabilities if requested
        probs = None
        if calc_probability:
            if probability_of_gt:
                sparse = self.to_sparse(batch.gtTexts)
            else:
                sparse = self.to_sparse(texts)
            
            loss_vals = tf.nn.ctc_loss(
                labels=sparse,
                logits=predictions,
                label_length=None,
                logit_length=seq_lengths,
                logits_time_major=False,
                blank_index=len(self.char_list)
            )
            probs = np.exp(-loss_vals.numpy())
        
        # Dump output if requested
        if self.dump:
            self.dump_nn_output(predictions.numpy())
        
        return texts, probs
    
    def dump_nn_output(self, rnn_output):
        """Dump the output of the NN to CSV files"""
        dump_dir = '../dump/'
        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)
        
        max_t, max_b, max_c = rnn_output.shape
        for b in range(max_b):
            csv = ''
            for t in range(max_t):
                for c in range(max_c):
                    csv += str(rnn_output[t, b, c]) + ';'
                csv += '\n'
            fn = dump_dir + 'rnnOutput_' + str(b) + '.csv'
            print('Write dump of NN to file: ' + fn)
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(csv)
    
    def save(self):
        """Save model weights"""
        self.snap_id += 1
        save_path = os.path.join(
            config.MODEL_PATH,
            f"{config.EXPERIMENT_NAME}_{self.snap_id}.weights.h5"
        )
        self.save_weights(save_path)
        print(f"Model saved to {save_path}")
    
    def _restore_model(self):
        """Restore model from checkpoint"""
        # Look for .weights.h5 files
        import glob
        checkpoint_pattern = os.path.join(config.MODEL_PATH, "*.weights.h5")
        checkpoints = glob.glob(checkpoint_pattern)
        
        if checkpoints:
            # Get the latest checkpoint
            latest = max(checkpoints, key=os.path.getctime)
            print(f'Restoring from {latest}')
            
            # Build model first by calling it with dummy data
            dummy_input = tf.zeros((1, config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
            _ = self(dummy_input, training=False)
            
            # Now load weights
            self.load_weights(latest)
            
            # Extract snap_id from filename
            import re
            match = re.search(r'_(\d+)\.weights\.h5$', latest)
            if match:
                self.snap_id = int(match.group(1))
        else:
            raise Exception(f'No saved model found in: {config.MODEL_PATH}')
    
    def get_model_summary(self):
        """Print model architecture summary"""
        print("\nModel Architecture Summary:")
        print("=" * 60)
        
        # Build model with sample input to get summary
        sample_input = tf.keras.Input(shape=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
        model = tf.keras.Model(inputs=sample_input, outputs=self.call(sample_input))
        model.summary()
        
        total_params = sum([tf.size(var).numpy() for var in self.trainable_variables])
        print(f"\nTotal trainable parameters: {total_params:,}")
        print("=" * 60)
