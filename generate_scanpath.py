import argparse
import torch
import numpy as np
from os.path import join
from models import Transformer
from CLIPGaze import CLIPGaze

def run_model(model, src, task, device="cuda:0", im_h=20, im_w=32, project_num=16):
    task = torch.tensor(task.astype(np.float32)).to(device).unsqueeze(0)
    firstfix = torch.tensor([(im_h//2)*project_num, (im_w//2)*project_num]).unsqueeze(0)

    with torch.no_grad():
        token_prob, ys, xs, ts = model(src=src, tgt=firstfix, task=task)

    token_prob = token_prob.cpu().numpy()
    ys = ys.cpu().numpy()
    xs = xs.cpu().numpy()
    ts = ts.cpu().numpy()

    ys_i = [(im_h//2)*project_num] + list(ys[:, 0, 0])[1:]
    xs_i = [(im_w//2)*project_num] + list(xs[:, 0, 0])[1:]
    ts_i = list(ts[:, 0, 0])
    token_type = [0] + list(np.argmax(token_prob[:, 0, :], axis=-1))[1:]

    scanpath = []
    for tok, y, x, t in zip(token_type, ys_i, xs_i, ts_i):
        if tok == 0:
            scanpath.append([y, x, t])
        else:
            break

    return np.array(scanpath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="checkpoint du modèle .pth")
    parser.add_argument("--image_ftrs", type=str, required=True, help="fichier .pth des features d'image")
    parser.add_argument("--task_emb", type=str, required=True, help="embedding numpy du task")
    parser.add_argument("--task", type=str, required=True, help="nom du task dans embedding")
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}")

    transformer = Transformer(
        nhead=8,
        d_model=256,
        num_decoder_layers=4,
        dim_feedforward=256,
        device=device
    ).to(device)

    model = CLIPGaze(
        transformer,
        spatial_dim=(20, 32),
        max_len=8,
        device=device
    ).to(device)

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    img_ftrs = [torch.load(args.image_ftrs, map_location=device)]
    embedding = np.load(args.task_emb, allow_pickle=True).item()
    task_vec = embedding[args.task]

    scanpath = run_model(model, img_ftrs, task_vec, device=device)
    print("Scanpath généré :")
    print(scanpath)

if __name__ == "__main__":
    main()
