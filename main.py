import json
import matplotlib.pyplot as plt


def main(log_path):

    with open(log_path, newline="", encoding="utf-8") as f:
        log_dict = json.load(f)

    prim_acc = log_dict['test']['prim_acc']
    plt.plot(prim_acc)
    plt.title('training 2')
    plt.show()

    pass


if __name__ == '__main__':
    main(r'C:\Users\xcheng\Desktop\attn_3dgcn_multitask_pmt_prim_cluster_2026-07-09_04-11-11.json')

