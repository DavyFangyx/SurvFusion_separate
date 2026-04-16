import argparse


DEFAULT_STUDY = 'tcga_kirc'
DEFAULT_RESULTS_DIR = './results'
DEFAULT_WSI_DIR = './SurvPGC_Workspace/P/uni_v1'
DEFAULT_CLINIC_DIR = './SurvPGC_Workspace/C/O_simple'
DEFAULT_GENE_DIR = './SurvPGC_Workspace/G/scFoundation_embedding_cell_norm'

def _process_args():
    r"""
    Function creates a namespace to read terminal-based arguments for running the experiment

    Args
        - None 

    Return:
        - args : argparse.Namespace

    """

    parser = argparse.ArgumentParser(description='Configurations for SurvPath Survival Prediction Training')

    #---> study related
    parser.add_argument('--study', type=str, default=DEFAULT_STUDY, help='study name')
    parser.add_argument('--task', type=str, default='survival', choices=['survival'])
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes (4 bins for survival)')
    parser.add_argument('--results_dir', default=DEFAULT_RESULTS_DIR, help='results directory (default: ./results)')
    parser.add_argument("--type_of_path", type=str, default="combine", choices=["xena", "hallmarks", "combine"])
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')

    #----> data related
    parser.add_argument('--data_root_dir', type=str, default=DEFAULT_WSI_DIR, help='data directory')
    parser.add_argument('--label_file', type=str, default=None, help='Path to csv with labels')
    parser.add_argument('--omics_dir', type=str, default=None, help='Path to dir with omics csv for all modalities')
    parser.add_argument('--clinic_dir', type=str, default=DEFAULT_CLINIC_DIR, help='Path to dir with clinical embedding')
    parser.add_argument('--gene_dir', type=str, default=DEFAULT_GENE_DIR, help='Path to dir with gene foundation model embedding')

    parser.add_argument('--num_patches', type=int, default=4096, help='number of patches')
    parser.add_argument('--label_col', type=str, default="survival_months", help='type of survival (OS, DSS, PFI, PFS)')
    parser.add_argument("--wsi_projection_dim", type=int, default=256)
    parser.add_argument("--encoding_layer_1_dim", type=int, default=8)
    parser.add_argument("--encoding_layer_2_dim", type=int, default=16)
    parser.add_argument("--encoder_dropout", type=float, default=0.25)

    #----> split related 
    parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--split_dir', type=str, default=None, help='manually specify the set of splits to use, '
                    +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--which_splits', type=str, default="5foldcv", help='where are splits')
        
    #----> training related 
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate (default: 0.0005)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--opt', type=str, default="radam", help="Optimizer")
    parser.add_argument('--reg_type', type=str, default="L2", help="regularization type [None, L1, L2]")
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--bag_loss', type=str, choices=['ce_surv', "nll_surv", "nll_rank_surv", "rank_surv", "cox_surv", "nll_diff_surv"], default='nll_surv',
                        help='survival loss function (default: ce)')
    parser.add_argument('--alpha_surv', type=float, default=0.5, help='weight given to uncensored patients')
    parser.add_argument('--beta_surv', type=float, default=0.3, help='weight given to diff_loss')
    parser.add_argument('--reg', type=float, default=0.001, help='weight decay / L2 (default: 0.001)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=1)

    #---> model related
    parser.add_argument('--fusion', type=str, default=None, choices=['concat', 'bilinear'])
    parser.add_argument('--modality', type=str, default="survpgc_f",
                        choices=['survpath', 'survpgc_f', 'omics', 'clinic_mlp', 'snn', 'clinic_snn', 'mlp_wsi', 'transmil_wsi', "abmil_wsi", 'survpc_f', 'survpc', 'coattn', 'porpoise', 'survpath_gc', 'mlppc_concat'])
    parser.add_argument('--return_attn', type=str, default=False, help="Used for heatmap drawing")
    parser.add_argument('--encoding_dim', type=int, default=1024, help='WSI encoding dim, default: 1024')
    parser.add_argument('--use_nystrom', action='store_true', default=False, help='Use Nystrom attentin in SurvPath.')

    args = parser.parse_args()

    if args.label_file is None:
        args.label_file = './datasets_csv/metadata/{}.csv'.format(args.study)

    if args.omics_dir is None:
        subtype = args.study.replace('tcga_', '')
        args.omics_dir = './datasets_csv/raw_rna_data/combine/{}'.format(subtype)

    if args.split_dir is None:
        args.split_dir = './splits/{}/{}'.format(args.which_splits, args.study)

    if not (args.task == "survival"):
        print("Task and folder does not match")
        exit()

    return args


'''
cd /data/fangyuxuan/projects/medical_dl/SurvPGC_github_init
conda activate SurvPGC

python main.py \
  --study tcga_kirc \
  --modality survpgc_f \
  --type_of_path combine \
  --k 5

  --fusion concat \
  --fusion bilinear \

'''


'''
# 单模态 G（只用基因组）
omics
snn

# 单模态 WSI（只用病理图像）
abmil_wsi
mlp_wsi
transmil_wsi

# 多模态基线（WSI + G）
porpoise
survpath
        # SurvPath：基因组 + 病理图像，以pathway为单位做注意力
        # 本项目最直接的前身方法，论文Table1中的重要对比基线

survpath_f（无）
        # SurvPath的foundation model版本，结构同上
        # _f 后缀表示使用了foundation model（如UNI/CONCH）编码的特征

# 本项目主模型（WSI + G + C）
survpgc_f
survfusion_f

'survfusion_mlpcontact', 
'survfusion_mhsaconcat', 
'survfusion_noalign'

# 消融：WSI + C（去掉G）
survpc
survpc_f

# 消融：WSI + C 直接拼接（无注意力）
mlppc_concat

# 消融：G + C（去掉WSI）
survgc_f

# 消融：只用C
clinic_mlp
clinic_snn
'''


