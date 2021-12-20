
# ====================
# Atari
# ====================

def add_atari_args(parser):
    parser.add_argument('--expt_name', type=str, required=True, help='experiment name')
    parser.add_argument('--task', type=str, required=True, choices=['train', 'record'], help='whether to train or record')
    parser.add_argument('--load', type=str, default=None, help='directory with model checkpoint to load')
    parser.add_argument('--wandb', action='store_true', help='log metrics on wandb')
    parser.add_argument('--double_dqn', action='store_true', help='have double dqn architecture for dqn agent')
    parser.add_argument('--dueling_dqn', action='store_true', help='have dueling dqn architecture for dqn agent')
    parser.add_argument('--frame_height', type=int, default=84, help='observation frame height')
    parser.add_argument('--frame_width', type=int, default=100, help='observation frame width')
    parser.add_argument('--frame_stack', type=int, default=4, help='frames to stack per observations')
    parser.add_argument('--frame_skip', type=int, default=4, help='frame skip, set None to avoid')
    parser.add_argument('--n_clusters', type=int, default=4, help='number of clusters')
    parser.add_argument('--reset_noops', type=int, default=30, help='random number of no-ops on reset upper limit')
    parser.add_argument('--episodic_life', type=bool, default=True, help='episodic life')
    parser.add_argument('--clip_rewards', type=bool, default=True, help='clip rewards by sign')
    parser.add_argument('--enc_hidden_ch', type=int, default=32, help='hidden conv layer channels in encoder')
    parser.add_argument('--enc_fdim', type=int, default=1024, help='encoder feature dim')
    parser.add_argument('--q_hidden_dim', type=int, default=512, help='q network mlp hidden dim')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor')
    parser.add_argument('--eps_max', type=float, default=1.0, help='exploration epsilon max value')
    parser.add_argument('--eps_min', type=float, default=0.01, help='exploration epsilon min value')
    parser.add_argument('--eps_decay_steps', type=int, default=500000, help='steps for linear epsilon decay')
    parser.add_argument('--entropy_weight_max', type=float, default=1.0, help='max entropy weight for clustering')
    parser.add_argument('--entropy_weight_min', type=float, default=0.01, help='min entropy weight for clustering')
    parser.add_argument('--entropy_decay_steps', type=int, default=500000, help='steps for linear entropy weight decay')
    parser.add_argument('--target_update_interval', type=int, default=10000, help='target net is updated every this many steps')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='network learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='gradient clipping max norm')
    parser.add_argument('--replay_mem_size', type=int, default=100000, help='replay memory size')
    parser.add_argument('--batch_size', type=int, default=32, help='replay memory sampling batch size')
    parser.add_argument('--cls_batch_size', type=int, default=256, help='replay memory sampling batch size for clustering')
    parser.add_argument('--mem_init_steps', type=int, default=50000, help='num random actions to initialize memory')
    parser.add_argument('--mem_replay_interval', type=int, default=4, help='memory replay happens every this many steps')
    parser.add_argument('--cls_learning_interval', type=int, default=4, help='clustering step happens every this many steps')
    parser.add_argument('--train_episodes', type=int, default=100000, help='num training episodes')
    parser.add_argument('--eval_episodes', type=int, default=10, help='num episodes in each evaluation cycle')
    parser.add_argument('--eval_interval', type=int, default=100, help='evaluation happens every this many episodes')
    parser.add_argument('--log_interval', type=int, default=1, help='logging to terminal happens every this many episodes')
    return parser

# ======================
# ViZDoom
# ======================

def add_vizdoom_args(parser):
    parser.add_argument('--expt_name', type=str, required=True, help='experiment name')
    parser.add_argument('--task', type=str, required=True, choices=['train', 'record'], help='whether to train or record')
    parser.add_argument('--load', type=str, default=None, help='directory with model checkpoint to load')
    parser.add_argument('--wandb', action='store_true', help='log metrics on wandb')
    parser.add_argument('--double_dqn', action='store_true', help='have double dqn architecture for dqn agent')
    parser.add_argument('--dueling_dqn', action='store_true', help='have dueling dqn architecture for dqn agent')
    parser.add_argument('--screen_format', type=str, default='GRAY8', help='vizdoom screen format')
    parser.add_argument('--screen_res', type=str, default='RES_640X480', help='vizdoom screen resolution')
    parser.add_argument('--frame_height', type=int, default=84, help='observation frame height')
    parser.add_argument('--frame_width', type=int, default=84, help='observation frame width')
    parser.add_argument('--frame_stack', type=int, default=3, help='frames to stack per observations')
    parser.add_argument('--frame_skip', type=int, default=12, help='frame skip, set None to avoid')
    parser.add_argument('--n_clusters', type=int, default=4, help='number of clusters')
    parser.add_argument('--enc_hidden_ch', type=int, default=32, help='hidden conv layer channels in encoder')
    parser.add_argument('--enc_fdim', type=int, default=1024, help='encoder feature dim')
    parser.add_argument('--q_hidden_dim', type=int, default=512, help='q network mlp hidden dim')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor')
    parser.add_argument('--eps_max', type=float, default=1.0, help='exploration epsilon max value')
    parser.add_argument('--eps_min', type=float, default=0.1, help='exploration epsilon min value')
    parser.add_argument('--eps_decay_steps', type=int, default=50000, help='steps for linear epsilon decay')
    parser.add_argument('--entropy_weight_max', type=float, default=1.0, help='max entropy weight for clustering')
    parser.add_argument('--entropy_weight_min', type=float, default=0.01, help='min entropy weight for clustering')
    parser.add_argument('--entropy_decay_steps', type=int, default=500000, help='steps for linear entropy weight decay')
    parser.add_argument('--target_update_interval', type=int, default=1, help='target net is updated every this many steps')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='network learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='gradient clipping max norm')
    parser.add_argument('--replay_mem_size', type=int, default=100000, help='replay memory size')
    parser.add_argument('--batch_size', type=int, default=32, help='replay memory sampling batch size')
    parser.add_argument('--cls_batch_size', type=int, default=256, help='replay memory sampling batch size for clustering')
    parser.add_argument('--mem_init_steps', type=int, default=5000, help='num random actions to initialize memory')
    parser.add_argument('--mem_replay_interval', type=int, default=1, help='memory replay happens every this many steps')
    parser.add_argument('--cls_learning_interval', type=int, default=1, help='clustering learning happens every this many steps')
    parser.add_argument('--train_episodes', type=int, default=5000, help='num training episodes')
    parser.add_argument('--eval_episodes', type=int, default=10, help='num episodes in each evaluation cycle')
    parser.add_argument('--spectator_episodes', type=int, default=10, help='num episodes to record videos for')
    parser.add_argument('--eval_interval', type=int, default=1000, help='evaluation happens every this many episodes')
    parser.add_argument('--log_interval', type=int, default=5, help='logging to terminal happens every this many episodes')
    return parser