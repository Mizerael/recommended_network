import matplotlib.pyplot as plt

def draw_barplor(df_x, df_y, title, xlabel, ylabel):
    plt.bar(df_x, df_y)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")

    plt.show()