# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import string

import torch

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import pandas as pd
from tqdm.notebook import tqdm
from Bio import SeqIO
import itertools
from typing import List, Tuple
import numpy as np
from matplotlib import pylab

AAorder=['K','R','H','E','D','N','Q','T','S','C','G','A','V','L','I','M','P','Y','F','W']

# xw
def generate_all_mutations(sequence, output_file):
    aas = 'ACDEFGHIKLMNPQRSTVWY'
    with open(output_file, 'w') as f:
        f.write('mutant\n')
        for i in range(len(sequence)):
            for aa in aas:
                f.write(sequence[i] + str(i + 1) + aa + '\n')
    f.close()


def generate_user_mutations(mutation, output_file):
    with open(output_file, 'w') as f:
        f.write('mutant\n')
        l = mutation.split(",")
        for i in l:
            f.write(i + '\n')
    f.close()

def generate_user_indels(sequence, indel, output_file):
    with open(output_file, 'w') as f:
        f.write(sequence + '\n')
        l = indel.split(",")
        for i in l:
            f.write(i + '\n')
    f.close()


# xw
def plot_clinvar(df, output_prefix, indel=False):
    pathogenic_scores = np.load('./ESM-Scan/pathogenic_scores.npy')
    benign_scores = np.load('./ESM-Scan/benign_scores.npy')
    pylab.rcParams['pdf.fonttype'] = 42
    pylab.rcParams['font.size'] = 12
    pylab.rcParams['figure.figsize'] = [10, 6]

    fig, ax = pylab.subplots()

    # Create histograms with density=True to normalize the data
    ax.hist(pathogenic_scores, bins=300, density=True, color='blue', alpha=0.5, label='ClinVar: Pathogenic')
    ax.hist(benign_scores, bins=300, density=True, color='red', alpha=0.5, label='Clinvar: Benign')

    if indel:
        value_to_plot = df.loc[0, '/content/esm1b_t33_650M_UR50S.pt']
        label = 'indel mutation'
        ax.axvline(x=value_to_plot, color='black', linestyle='--', alpha=0.5)
        ax.text(value_to_plot, 0.01, label, rotation=90, verticalalignment='bottom', horizontalalignment='right')
    else:
        values_to_plot = df['/content/esm1b_t33_650M_UR50S.pt']
        labels = df['mutant']
        for label, value in zip(labels, values_to_plot):
            ax.axvline(x=value, color='black', linestyle='--', alpha=0.5)  # Customize the line style and transparency
            ax.text(value, 0.01, label, rotation=90, verticalalignment='bottom', horizontalalignment='right')

    # Add labels and legend
    ax.set_xlabel('Scores')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')

    # Show the plot
    ax.set_title('Pathogenic vs. Benign Score Density Plot and Histogram')
    pylab.close()
    fig.savefig(str(output_prefix) + '-densityplot.pdf')


def plot_esm_scan(seq, output_prefix):
    # read output in a list format
    df = pd.read_csv(str(output_prefix) + '-res-in-list.csv')

    # convert to a matrix, each row is a wild type aa/position in the sequence, each column is a mutation
    d = np.array(df[df.columns[1]]).reshape(len(seq), 20)

    # convert it to a data frame
    df2 = pd.DataFrame(d, columns=[aa for aa in 'ACDEFGHIKLMNPQRSTVWY'],
                       index=[seq[x] + str(x + 1) for x in range(len(seq))])
    df2.to_csv(str(args.output_prefix) + '-res-in-matrix.csv')

    # plot matrix as heatmap
    pylab.rcParams['pdf.fonttype'] = 42
    pylab.rcParams['font.size'] = 12
    pylab.rcParams['figure.figsize'] = [8, len(seq) / 3]

    fig, ax = pylab.subplots()
    im = ax.imshow(d, vmin=np.min(d), vmax=np.max(d), cmap='bwr', aspect='equal')

    ax.set_ylabel('Wild type')
    yticks = list(range(len(seq)))
    ax.set_yticks(yticks)
    ax.set_yticklabels(seq[x] + str(x + 1) for x in range(len(seq)))

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Mutant')
    xticks = list(range(20))
    ax.set_xticks(xticks)
    ax.set_xticklabels(aa for aa in 'ACDEFGHIKLMNPQRSTVWY')

    # colorbar
    fig.colorbar(im, orientation="horizontal")

    fig.savefig(str(output_prefix) + '-matrix.pdf')

    # boxplot
    pylab.close()
    pylab.rcParams['pdf.fonttype'] = 42
    pylab.rcParams['font.size'] = 12
    pylab.rcParams['figure.figsize'] = [8, len(seq) / 3]
    boxplot = df2[::-1].T.boxplot(grid=False, vert=False)
    pylab.savefig(str(output_prefix) + '-boxplot.pdf')


def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None

    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)


def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions.

    The input file must be in a3m format (although we use the SeqIO fasta parser)
    for remove_insertions to work properly."""

    msa = [
        (record.description, remove_insertions(str(record.seq)))
        for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
    ]
    return msa


def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from an ensemble of ESM-1v models."  # noqa
    )

    # fmt: off
    parser.add_argument(
        "--model-location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
        nargs="+",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Base sequence to which mutations were applied",
    )
    parser.add_argument(
        "--dms-input",
        type=pathlib.Path,
        help="CSV file containing the deep mutational scan",
    )
    parser.add_argument(
        "--dms-mutation",
        type=str,
        help="comma separated str of different mutations",
    )
    parser.add_argument(
        "--dms-indel",
        type=str,
        help="comma separated str specifying indel",
    )
    parser.add_argument(
        "--mutation-col",
        type=str,
        default="mutant",
        help="column in the deep mutational scan labeling the mutation as 'AiB'"
    )
    parser.add_argument(
        "--output-prefix",
        type=pathlib.Path,
        default="ESMScan",
        help="Output file containing the deep mutational scan along with predictions",
    )
    parser.add_argument(
        "--offset-idx",
        type=int,
        default=1,  # xw, changed from 0 to 1
        help="Offset of the mutation positions in `--mutation-col`"
    )
    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="wt-marginals",
        choices=["wt-marginals", "masked-marginals", "indel"],
        help=""
    )
    parser.add_argument(
        "--msa-path",
        type=pathlib.Path,
        help="path to MSA in a3m format (required for MSA Transformer)"
    )
    parser.add_argument(
        "--msa-samples",
        type=int,
        default=400,
        help="number of sequences to select from the start of the MSA"
    )
    # fmt: on
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def label_row(row, sequence, token_probs, alphabet, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # add 1 for BOS
    score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
    return score.item()

def get_logits(seq,model,alphabet,batch_converter,format=None):
  data = [ ("_", seq),]
  batch_labels, batch_strs, batch_tokens = batch_converter(data)
  batch_tokens = batch_tokens.cuda()
  with torch.no_grad():
      logits = torch.log_softmax(model(batch_tokens, repr_layers=[33], return_contacts=False)["logits"],dim=-1).cpu().numpy()
  if format=='pandas':
    WTlogits = pd.DataFrame(logits[0][1:-1,:],columns=alphabet.all_toks,index=list(seq)).T.iloc[4:24].loc[AAorder]
    WTlogits.columns = [j.split('.')[0]+' '+str(i+1) for i,j in enumerate(WTlogits.columns)]
    return WTlogits
  else:
    return logits[0][1:-1,:]

def compute_pppl(row, sequence, model, alphabet, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    # modify the sequence
    sequence = sequence[:idx] + mt + sequence[(idx + 1):]

    # encode the sequence
    data = [
        ("protein1", sequence),
    ]

    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # compute probabilities at each position
    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
        log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())  # vocab size
    return log_probs

def get_PLL(seq,model,alphabet,batch_converter,reduce=np.sum):
  s=get_logits(seq,alphabet=alphabet,model=model, batch_converter=batch_converter)
  idx=[alphabet.tok_to_idx[i] for i in seq]
  return reduce(np.diag(s[:,idx]))


def get_PLLR(wt_seq,mut_seq,start_pos,model,alphabet,batch_converter,weighted=False):
  fn=np.sum if not weighted else np.mean
  if max(len(wt_seq),len(mut_seq))<=1022:
    return  get_PLL(mut_seq,model=model,alphabet=alphabet,batch_converter=batch_converter,reduce=fn) - get_PLL(wt_seq,model=model,alphabet=alphabet,batch_converter=batch_converter,reduce=fn)
  else:
    wt_seq,mut_seq,start_pos = crop_indel(wt_seq,mut_seq,start_pos)
    return  get_PLL(mut_seq,model=model,alphabet=alphabet,batch_converter=batch_converter,reduce=fn) - get_PLL(wt_seq,model=model,alphabet=alphabet,batch_converter=batch_converter,reduce=fn)


def crop_indel(ref_seq, alt_seq, ref_start):
    # Start pos: 1-indexed start position of variant
    left_pos = ref_start - 1
    offset = len(ref_seq) - len(alt_seq)
    start_pos = int(left_pos - 1022 / 2)
    end_pos1 = int(left_pos + 1022 / 2) - min(start_pos, 0) + min(offset, 0)
    end_pos2 = int(left_pos + 1022 / 2) - min(start_pos, 0) - max(offset, 0)
    if start_pos < 0: start_pos = 0  # Make sure the start position is not negative
    if end_pos1 > len(ref_seq): end_pos1 = len(
        ref_seq)  # Make sure the end positions are not beyond the end of the sequence
    if end_pos2 > len(alt_seq): end_pos2 = len(alt_seq)
    if start_pos > 0 and max(end_pos2, end_pos1) - start_pos < 1022:  ## extend to the left if there's space
        start_pos = max(0, max(end_pos2, end_pos1) - 1022)

    return ref_seq[start_pos:end_pos1], alt_seq[start_pos:end_pos2], start_pos - ref_start


def main(args):
    if args.dms_mutation:
        args.dms_input = str(args.output_prefix) + '-user-mutants.txt'
        generate_user_mutations(args.dms_mutation, args.dms_input)
    elif args.dms_indel:
        args.dms_input = str(args.output_prefix) + '-user-indel.txt'
        generate_user_indels(args.sequence, args.dms_indel, args.dms_input)
    else: # generate all possible mutants
        args.dms_input = str(args.output_prefix) + '-all-mutants.txt'
        generate_all_mutations(args.sequence, args.dms_input)
        # Load the deep mutational scan
    df = pd.read_csv(args.dms_input)

    # inference for each model
    for model_location in args.model_location:
        model, alphabet = pretrained.load_model_and_alphabet(model_location)
        model.eval()

        if torch.cuda.is_available() and not args.nogpu:
            model = model.cuda()
            print("Transferred model to GPU")

        batch_converter = alphabet.get_batch_converter()

        if isinstance(model, MSATransformer):
            data = [read_msa(args.msa_path, args.msa_samples)]
            assert (
                    args.scoring_strategy == "masked-marginals"
            ), "MSA Transformer only supports masked marginal strategy"

            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            all_token_probs = []
            for i in tqdm(range(batch_tokens.size(2))):
                batch_tokens_masked = batch_tokens.clone()
                batch_tokens_masked[0, 0, i] = alphabet.mask_idx  # mask out first sequence
                with torch.no_grad():
                    token_probs = torch.log_softmax(
                        model(batch_tokens_masked.cuda())["logits"], dim=-1
                    )
                all_token_probs.append(token_probs[:, 0, i])  # vocab size
            token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
            df[model_location] = df.apply(
                lambda row: label_row(
                    row[args.mutation_col], args.sequence, token_probs, alphabet, args.offset_idx
                ),
                axis=1,
            )

        else:
            data = [
                ("protein1", args.sequence),
            ]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            if args.scoring_strategy == "wt-marginals":
                with torch.no_grad():
                    token_probs = torch.log_softmax(model(batch_tokens.cuda())["logits"], dim=-1)
                df[model_location] = df.apply(
                    lambda row: label_row(
                        row[args.mutation_col],
                        args.sequence,
                        token_probs,
                        alphabet,
                        args.offset_idx,
                    ),
                    axis=1,
                )
            elif args.scoring_strategy == "masked-marginals":
                all_token_probs = []
                for i in tqdm(range(batch_tokens.size(1))):
                    batch_tokens_masked = batch_tokens.clone()
                    batch_tokens_masked[0, i] = alphabet.mask_idx
                    with torch.no_grad():
                        token_probs = torch.log_softmax(
                            model(batch_tokens_masked.cuda())["logits"], dim=-1
                        )
                    all_token_probs.append(token_probs[:, i])  # vocab size
                token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
                df[model_location] = df.apply(
                    lambda row: label_row(
                        row[args.mutation_col],
                        args.sequence,
                        token_probs,
                        alphabet,
                        args.offset_idx,
                    ),
                    axis=1,
                )
            elif args.scoring_strategy == "indel":
                tqdm.pandas()
                df = df.transpose()
                df.reset_index(inplace=True)
                df.columns = ['wt_seq','mut_seq','start_pos']
                df[model_location] = df.progress_apply(
                    lambda row: get_PLLR(
                        row['wt_seq'], row['mut_seq'], row['start_pos'], model, alphabet, batch_converter
                    ),
                    axis = 1,
                )

    df.to_csv(str(args.output_prefix) + '-res-in-list.csv', index=False)

    # xw: plot
    if args.dms_mutation:
        plot_clinvar(df, args.output_prefix,indel=False)
    elif args.dms_indel:
        plot_clinvar(df, args.output_prefix,indel=True)
    else:
        plot_esm_scan(args.sequence, args.output_prefix)



if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
