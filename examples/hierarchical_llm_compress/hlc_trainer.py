"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import time
import evaluate

from graphstorm.trainer import GSgnnTrainer

"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Customized huggingface trainer for GraphStorm
"""

from transformers import Trainer

class GsHuggingfaceTrainer(Trainer):
    """ Customize Huggingface Trainer
    """
    def __init__(self, train_loader, val_loader, test_loader, **kwargs):
        self._train_dataloader = train_loader
        self._val_dataloader = val_loader
        self._test_dataloader = test_loader
        super(GsHuggingfaceTrainer, self).__init__(**kwargs)

    def get_train_dataloader(self):
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return self.accelerator.prepare(self._train_dataloader)

    def get_eval_dataloader(self, eval_dataset=None):
        return self.accelerator.prepare(self._val_dataloader)

    def get_test_dataloader(self, test_dataset=None):
        return self.accelerator.prepare(self._test_dataloader)

class HLCNodePredictionTrainer(GSgnnTrainer):
    """ A trainer for node prediction with
        hierarchical lm compressor.

        Parameters
        ----------
        model: HLCNodeModel
            The hierarchical lm compressor model for node prediction.
    """
    def __init__(self, model):
        self._model = model

    def fit(self, train_dataset,
            training_args,
            train_loader,
            compute_metrics,
            val_loader=None,
            test_loader=None):
        """ The fit function for node prediction.

        Parameters
        ----------
        train_dataset:

        train_loader : GSgnnLinkPredictionDataLoader
            The mini-batch sampler for training.
        training_args: dict
            Args for Huggingface trainer
        train_loader : GSlmHatNodeDataLoader
            The mini-batch sampler for training.
        compute_metrics: func
            Function to compute metrics
        val_loader : GSlmHatNodeDataLoader
            The mini-batch sampler for computing validation scores. The validation scores
            are used for selecting models.
        test_loader : GSlmHatNodeDataLoader
            The mini-batch sampler for computing test scores.
        """
        trainer = GsHuggingfaceTrainer(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None, # GraphStorm store eval and test set in train_dataset
            compute_metrics=compute_metrics)

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint

            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model(training_args.output_dir)  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)