import pandas as pd
import matplotlib.pyplot as plt


def compute_ranking_loss_df(df):
    """
    Returns a new DataFrame with an additional 'loss' column.
    Assumes Yagiz and Tugba lists are ordered song indices ranked from 1 to 5.
    """
    loss_list = []

    for _, row in df.iterrows():
        y_list = row['Yagiz']
        t_list = row['Tugba']

        common = set(y_list) & set(t_list)
        k = len(common)

        # Loss for songs not in common
        top5_loss = 10 * (5 - k)

        # Rank dictionaries
        y_ranks = {idx: rank for rank, idx in enumerate(y_list, start=1)}
        t_ranks = {idx: rank for rank, idx in enumerate(t_list, start=1)}

        # Rank difference for shared songs
        rank_diff_loss = sum(abs(y_ranks[i] - t_ranks[i]) for i in common)

        total_loss = top5_loss + rank_diff_loss
        loss_list.append(total_loss)

    # Return new dataframe with an extra column
    df_with_loss = df.copy()
    df_with_loss['loss'] = loss_list
    return df_with_loss

def plot_ranking_losses(ranking_loss_df):
    losses = ranking_loss_df['loss'].tolist()
    album_names = ranking_loss_df['album'].tolist()

    plt.figure(figsize=(10, 5))
    plt.plot(album_names, losses, marker='o', label='Loss', linewidth = 2)
    plt.title('Disagreement Loss per Album')
    plt.xlabel('Album Name')
    plt.ylabel('Loss')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

