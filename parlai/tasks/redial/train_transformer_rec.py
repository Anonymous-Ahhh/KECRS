#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Train a model using parlai's standard training loop.

For documentation, see parlai.scripts.train_model.
"""

import sys
sys.path.append("../../../")
from parlai.scripts.train_model import TrainLoop, setup_args, _maybe_load_eval_world, run_eval
from parlai.core.agents import create_agent

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task='redial',
        model='transformer_rec/generator',
        model_file='saved_orig2/transformer_rec',
        dict_tokenizer='nltk',
        dict_lower=True,
        batchsize=1,
        truncate=512,
        dropout=0.1,
        relu_dropout=0.1,
        n_entity=64368,
        n_relation=214,
        validation_metric='nll_loss',
        validation_metric_mode='min',
        validation_every_n_secs=300,
        validation_patience=5,
        tensorboard_log=True,
        tensorboard_tag="task,model,batchsize,ffn_size,embedding_size,n_layers,learningrate,model_file",
        tensorboard_metrics="ppl,nll_loss,token_acc,bleu",
    )
    opt = parser.parse_args()
    # TrainLoop(opt).train()
    agent = create_agent(opt)
    test_world = _maybe_load_eval_world(agent, opt, 'test')
    t_report = run_eval(test_world, opt, 'test', -1, write_log=True)
