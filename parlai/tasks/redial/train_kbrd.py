#!/data/qibin/anaconda3/envs/alchemy/bin/python

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


if __name__ == "__main__":
    parser = setup_args()
    parser.set_defaults(
        task="redial",
        dict_tokenizer="split",
        model="kbrd",
        # datatype='test',
        dict_file="saved_old_setting/tmp",
        model_file="saved_old_setting/kbrd",
        fp16=True,
        batchsize=64,
        n_entity=64368,
        n_relation=214,
        # validation_metric="recall@50",
        validation_metric="base_loss",
        validation_metric_mode='min',
        # validation_every_n_epochs=1,
        validation_every_n_secs=30,
        validation_patience=5,
        tensorboard_log=True,
        tensorboard_tag="task,model,batchsize,dim,learningrate,model_file",
        tensorboard_metrics="loss,base_loss,kge_loss,l2_loss,acc,auc,recall@1,recall@10,recall@50",
        sub_graph=True,
        hop=2,
        L=1
    )
    opt = parser.parse_args()
    # TrainLoop(opt).train()
    agent = create_agent(opt)
    test_world = _maybe_load_eval_world(agent, opt, 'test')
    t_report = run_eval(test_world, opt, 'test', -1, write_log=True)
    # print(1)
