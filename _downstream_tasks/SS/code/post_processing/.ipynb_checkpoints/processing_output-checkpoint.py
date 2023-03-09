import numpy as np
import pandas as pd
import os, six, sys, subprocess, time


def prob_output(pred_data, pred_path):
    """
    INPUT: a tuple of (predict, label)
    output prob_file
    """
    pred_label = pred_data
    for id, seq, y_pred in pred_label:
        y_pred = y_pred.cpu().numpy()  # GPU data to CPU
        y_pred = np.squeeze(y_pred)
        pred_file = pred_path + "{}.prob".format(id)
        np.savetxt(pred_file, y_pred)  # .numpy())
    return (id, seq, y_pred)


def ct_file_output(pairs, seq, id, save_result_path):
    col1 = np.arange(1, len(seq) + 1, 1)
    col2 = np.array([i for i in seq])
    col3 = np.arange(0, len(seq), 1)
    col4 = np.append(np.delete(col1, 0), [0])
    col5 = np.zeros(len(seq), dtype=int)

    for i, I in enumerate(pairs):
        col5[I[0]] = int(I[1]) + 1
        col5[I[1]] = int(I[0]) + 1
    col6 = np.arange(1, len(seq) + 1, 1)
    temp = np.vstack((np.char.mod('%d', col1), col2, np.char.mod('%d', col3), np.char.mod('%d', col4),
                      np.char.mod('%d', col5), np.char.mod('%d', col6))).T
    # os.chdir(save_result_path)
    # print(os.path.join(save_result_path, str(id[0:-1]))+'.spotrna')
    np.savetxt(os.path.join(save_result_path, str(id)) + '.ct', (temp), delimiter='\t\t', fmt="%s",
               header=str(len(seq)) + '\t\t' + str(id) + '\t\t' + 'SPOT-RNA output\n', comments='')

    return


def bpseq_file_output(pairs, seq, id, save_result_path):
    col1 = np.arange(1, len(seq) + 1, 1)
    col2 = np.array([i for i in seq])
    # col3 = np.arange(0, len(seq), 1)
    # col4 = np.append(np.delete(col1, 0), [0])
    col5 = np.zeros(len(seq), dtype=int)

    for i, I in enumerate(pairs):
        col5[I[0]] = int(I[1]) + 1
        col5[I[1]] = int(I[0]) + 1
    # col6 = np.arange(1, len(seq) + 1, 1)
    temp = np.vstack((np.char.mod('%d', col1), col2, np.char.mod('%d', col5))).T
    # os.chdir(save_result_path)
    # print(os.path.join(save_result_path, str(id[0:-1]))+'.spotrna')
    np.savetxt(os.path.join(save_result_path, str(id)) + '.bpseq', (temp), delimiter=' ', fmt="%s",
               header='#' + str(id), comments='')

    return


def output_mask(seq, NC=True):
    if NC:
        include_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG', 'CC', 'GG', 'AG', 'CA', 'AC', 'UU', 'AA', 'CU', 'GA', 'UC']
    else:
        include_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']
    mask = np.zeros((len(seq), len(seq)))
    for i, I in enumerate(seq):
        for j, J in enumerate(seq):
            if str(I) + str(J) in include_pairs:
                mask[i, j] = 1
    return mask


def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def multiplets_pairs(pred_pairs):
    pred_pair = [i[:2] for i in pred_pairs]
    temp_list = flatten(pred_pair)
    temp_list.sort()
    new_list = sorted(set(temp_list))
    dup_list = []
    for i in range(len(new_list)):
        if (temp_list.count(new_list[i]) > 1):
            dup_list.append(new_list[i])

    dub_pairs = []
    for e in pred_pair:
        if e[0] in dup_list:
            dub_pairs.append(e)
        elif e[1] in dup_list:
            dub_pairs.append(e)

    temp3 = []
    for i in dup_list:
        temp4 = []
        for k in dub_pairs:
            if i in k:
                temp4.append(k)
        temp3.append(temp4)

    return temp3


def multiplets_free_bp(pred_pairs, y_pred):
    L = len(pred_pairs)
    multiplets_bp = multiplets_pairs(pred_pairs)
    save_multiplets = []
    while len(multiplets_bp) > 0:
        remove_pairs = []
        for i in multiplets_bp:
            save_prob = []
            for j in i:
                save_prob.append(y_pred[j[0], j[1]])
            remove_pairs.append(i[save_prob.index(min(save_prob))])
            save_multiplets.append(i[save_prob.index(min(save_prob))])
        pred_pairs = [k for k in pred_pairs if k not in remove_pairs]
        multiplets_bp = multiplets_pairs(pred_pairs)
    save_multiplets = [list(x) for x in set(tuple(x) for x in save_multiplets)]
    assert L == len(pred_pairs) + len(save_multiplets)
    # print(L, len(pred_pairs), save_multiplets)
    return pred_pairs, save_multiplets


def type_pairs(pairs, sequence):
    sequence = [i.upper() for i in sequence]
    # seq_pairs = [[sequence[i[0]],sequence[i[1]]] for i in pairs]

    AU_pair = []
    GC_pair = []
    GU_pair = []
    other_pairs = []
    for i in pairs:
        if [sequence[i[0]], sequence[i[1]]] in [["A", "U"], ["U", "A"]]:
            AU_pair.append(i)
        elif [sequence[i[0]], sequence[i[1]]] in [["G", "C"], ["C", "G"]]:
            GC_pair.append(i)
        elif [sequence[i[0]], sequence[i[1]]] in [["G", "U"], ["U", "G"]]:
            GU_pair.append(i)
        else:
            other_pairs.append(i)
    watson_pairs_t = AU_pair + GC_pair
    wobble_pairs_t = GU_pair
    other_pairs_t = other_pairs
    # print(watson_pairs_t, wobble_pairs_t, other_pairs_t)
    return watson_pairs_t, wobble_pairs_t, other_pairs_t


def lone_pair(pairs):
    lone_pairs = []
    pairs.sort()
    for i, I in enumerate(pairs):
        if ([I[0] - 1, I[1] + 1] not in pairs) and ([I[0] + 1, I[1] - 1] not in pairs):
            lone_pairs.append(I)

    return lone_pairs


def prob_to_secondary_structure(model_outputs, seq, name, args, base_path, output_dir):
    # save_result_path = 'outputs'
    Threshold = 0.516
    y_pred = model_outputs
    if y_pred.is_cuda:
       y_pred = y_pred.cpu().numpy()

    """
    test_output = ensemble_outputs
    mask = output_mask(seq)
    inds = np.where(label_mask == 1)
    y_pred = np.zeros(label_mask.shape)
    for i in range(test_output.shape[0]):
        y_pred[inds[0][i], inds[1][i]] = test_output[i]
    y_pred = np.multiply(y_pred, mask)
    """

    tri_inds = np.triu_indices(y_pred.shape[0], k=1)
    out_pred = y_pred[tri_inds]
    upper_list = [[i, j, y_pred[i, j]] for i, j in zip(tri_inds[0], tri_inds[1])]
    pred_pairs = [[i, j] for i, j, prob in upper_list if prob > Threshold]
    pred_pairs, save_multiplets = multiplets_free_bp(pred_pairs, y_pred)

    watson_pairs, wobble_pairs, noncanonical_pairs = type_pairs(pred_pairs, seq)
    lone_bp = lone_pair(pred_pairs)

    tertiary_bp = save_multiplets + noncanonical_pairs + lone_bp
    tertiary_bp = [list(x) for x in set(tuple(x) for x in tertiary_bp)]

    str_tertiary = []
    for i, I in enumerate(tertiary_bp):
        if i == 0:
            str_tertiary += ('(' + str(I[0] + 1) + ',' + str(I[1] + 1) + '):color=""#FFFF00""')
        else:
            str_tertiary += (';(' + str(I[0] + 1) + ',' + str(I[1] + 1) + '):color=""#FFFF00""')

    tertiary_bp = ''.join(str_tertiary)


    output_path = os.path.join(output_dir, 'SS_result')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ct_file_output(pred_pairs, seq, name, output_path)
    bpseq_file_output(pred_pairs, seq, name, output_path)
    np.savetxt(output_path + '/' + name + '.prob', y_pred, delimiter='\t')

    if args.plots:
        print(f'tertiary_bp={tertiary_bp}')
        VARNA = os.path.join(base_path, 'tools/VARNAv3-93.jar')
        plot_input = os.path.join(output_path, name + '.ct')
        rad_output = os.path.join(output_path, name + '_radiate.png')
        lin_output = os.path.join(output_path, name + '_line.png')
        # print(f'cwd={os.path.abspath(os.path.curdir)}')
        # print(f'VARNA={VARNA}')
        # print(f'plot_input={plot_input}')
        # print(f'rad_output={rad_output}')
        # print(f'lin_output={lin_output}', flush=True)
        # subprocess.Popen(['java', '-version'])
        sp1_std = subprocess.Popen(
            ["java", "-cp", VARNA, "fr.orsay.lri.varna.applications.VARNAcmd", '-i', plot_input, '-o', rad_output,
             '-algorithm', 'radiate', '-resolution', '8.0', '-bpStyle', 'lw', '-auxBPs', tertiary_bp],
            stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
        print(sp1_std.decode())
        sp2_std = subprocess.Popen(
            ["java", "-cp", VARNA, "fr.orsay.lri.varna.applications.VARNAcmd", '-i', plot_input, '-o', lin_output,
             '-algorithm', 'line', '-resolution', '8.0', '-bpStyle', 'lw', '-auxBPs', tertiary_bp],
            stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
        print(sp2_std.decode(), flush=True)
    # except:
    #    print('\nUnable to generate 2D plots;\nplease refer to "http://varna.lri.fr/" for system requirments to use VARNA')
    """
    if args.motifs:
        try:
            os.chdir(output_path)
            p = subprocess.Popen(['perl', base_path + '/utils/bpRNA-master/bpRNA.pl', name + '.bpseq']).wait()
            time.sleep(0.1)
        except:
            print(
                '\nUnable to run bpRNA script;\nplease refer to "https://github.com/hendrixlab/bpRNA/" for system requirments to use bpRNA')
            os.chdir(base_path)
        # print(os.path.exists(output_path + '/' + name + '.st'))
        print(f'cwd={os.path.abspath(os.path.curdir)}')
        with open(name + '.st') as f:
            df = pd.read_csv(f, comment='#', sep=";", header=None)
        np.savetxt(name + '.dbn', np.array([df[0][0], df[0][1]]), fmt="%s", header='>' + name, comments='')
        os.chdir(base_path)
    """
