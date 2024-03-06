import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def save_fig_png(fig, filepath: str) -> None:
    fig.savefig(filepath, bbox_inches='tight', dpi=600)


def exp1_figures():
    data = loadmat("paper_results/exp1.mat")
    
    snr = data["snr"].T
    cResFreq = data["PSNR_cresfreq"].T
    SwinFreq = data["PSNR_swinfreq"].T
    CVSwinFreq = data["PSNR_cvswinfreq"].T
    
    fig1, ax1 = create_figure_exp1(snr, cResFreq, SwinFreq, CVSwinFreq)
    ax1.set_ylabel("PSNR")
    
    save_fig_png(fig1, "paper_results/saved_figures/py/exp1_psnr.png")
    
    cResFreq = data["SSIM_cresfreq"].T
    SwinFreq = data["SSIM_swinfreq"].T
    CVSwinFreq = data["SSIM_cvswinfreq"].T
    
    fig2, ax2 = create_figure_exp1(snr, cResFreq, SwinFreq, CVSwinFreq)
    ax2.set_ylabel("SSIM")
    
    save_fig_png(fig2, "paper_results/saved_figures/py/exp1_ssim.png")
    
    return (fig1, fig2), (ax1, ax2)
    
    
def create_figure_exp1(snr, cResFreq, SwinFreq, CVSwinFreq):
    linewidth = 2
    fmt = "-o"
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    
    fig, ax = plt.subplots()
    ax.plot(snr, cResFreq, fmt, linewidth=linewidth)
    ax.plot(snr, SwinFreq, fmt, linewidth=linewidth)
    ax.plot(snr, CVSwinFreq, fmt, linewidth=linewidth)
    ax.legend(["cResFreq", "SwinFreq", "CVSwinFreq"], loc=4)
    ax.grid()
    ax.set_xlabel("SNR (dB)")
    
    return fig, ax


def exp2_figure():
    linewidth = 2
    fmt = "-"
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    
    data = loadmat("paper_results/exp2.mat")
    
    sep = data["seperation"].T
    Periodogram = data["Periodogram"].T
    MUSIC = data["MUSIC"].T
    FISTA = data["FISTA"].T
    cResFreq = data["cresfreq"].T
    SwinFreq = data["swinfreq"].T
    CVSwinFreq = data["cvswinfreq"].T
    
    fig, ax = plt.subplots()
    ax.plot(sep, Periodogram, fmt, linewidth=linewidth)
    ax.plot(sep, MUSIC, fmt, linewidth=linewidth)
    ax.plot(sep, FISTA, fmt, linewidth=linewidth)
    ax.plot(sep, cResFreq, fmt, linewidth=linewidth)
    ax.plot(sep, SwinFreq, fmt, linewidth=linewidth)
    ax.plot(sep, CVSwinFreq, fmt, linewidth=linewidth)
    ax.legend([
        "Periodogram",
        "MUSIC",
        "FISTA",
        "cResFreq",
        "SwinFreq",
        "CVSwinFreq"
    ], fontsize=12, loc=4)
    ax.grid()
    ax.set_xlabel("Frequency Separation Interval")
    ax.set_ylabel("Probability")
    ax.set_xlim(sep[0], 1)
    
    save_fig_png(fig, "paper_results/saved_figures/py/exp2_res.png")
    
    return fig, ax
    

def exp3_figures():
    data = loadmat("paper_results/exp3.mat")
    
    snr = data["snr"]
    freqs = data["f"]
        
    Periodogram = data["Periodogram"]
    MUSIC = data["MUSIC"]
    FISTA = data["FISTA"]
    cResFreq = data["cresfreq"]
    SwinFreq = data["swinfreq"]
    CVSwinFreq = data["cvswinfreq"]
    
    fig_all = []
    ax_all = []
    
    for i_snr, snr_i in enumerate(snr):
        for i_sample in range(cResFreq.shape[1]):
            fig, ax = plt.subplots()
            
            Periodogram_i = Periodogram[:, i_sample, i_snr]
            MUSIC_i = MUSIC[:, i_sample, i_snr]
            FISTA_i = FISTA[:, i_sample, i_snr]
            cResFreq_i = cResFreq[:, i_sample, i_snr]
            SwinFreq_i = SwinFreq[:, i_sample, i_snr]
            CVSwinFreq_i = CVSwinFreq[:, i_sample, i_snr]
            
            f_i = freqs[i_sample, :, i_snr]
            f_i = f_i[f_i != -10]
            
            if snr_i == 0:
                mindB = -60
            elif snr_i == 20:
                mindB = -80
                
            create_figure_exp3(
                ax,
                Periodogram_i,
                MUSIC_i,
                FISTA_i,
                cResFreq_i,
                SwinFreq_i,
                CVSwinFreq_i,
                f_i,
                mindB
            )
            
            save_fig_png(fig, f"paper_results/saved_figures/py/exp3_SNR{snr_i}dB_{i_sample}")
            
            fig_all.append(fig)
            ax_all.append(ax)
            
    return fig_all, ax_all
            


def create_figure_exp3(
    ax,
    Periodogram,
    MUSIC,
    FISTA,
    cResFreq,
    SwinFreq,
    CVSwinFreq,
    freqs,
    mindB
):
    linewidth = 1
    fmt = "-"
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    
    def normalize_exp3(x, lower_thr=-np.inf):
        x[x < np.finfo(float).eps] = np.finfo(float).eps
        y = 20 * np.log10(x / np.max(x))
        y[y < lower_thr] = lower_thr - 10
        return y
    
    Periodogram = normalize_exp3(Periodogram)
    MUSIC = normalize_exp3(MUSIC)
    FISTA = normalize_exp3(FISTA)
    cResFreq = normalize_exp3(cResFreq)
    SwinFreq = normalize_exp3(SwinFreq)
    CVSwinFreq = normalize_exp3(CVSwinFreq)
    
    f_axis = np.linspace(-0.5, 0.5, len(Periodogram), endpoint=True)
    
    ax.plot(f_axis, Periodogram, fmt, linewidth=linewidth)
    ax.plot(f_axis, MUSIC, fmt, linewidth=linewidth)
    ax.plot(f_axis, FISTA, fmt, linewidth=linewidth)
    ax.plot(f_axis, cResFreq, fmt, linewidth=linewidth)
    ax.plot(f_axis, SwinFreq, fmt, linewidth=linewidth)
    ax.plot(f_axis, CVSwinFreq, fmt, linewidth=linewidth)
    
    for freq_i in freqs:
        ax.axvline(freq_i, linestyle="--", linewidth=3, color="r")
    
    ax.axhline(0, color="k")
    ax.legend([
        "Periodogram",
        "MUSIC",
        "FISTA",
        "cResFreq",
        "SwinFreq",
        "CVSwinFreq"
    ], fontsize=12, loc=3)
    ax.grid()
    ax.set_xlabel("f / Hz")
    ax.set_ylabel("Normalized Power / dB")
    
    ax.set_xlim(
        np.round(np.min(freqs) - 0.02,2),
        np.round(np.max(freqs) + 0.02,2)
    )
    ax.set_ylim(mindB, 0)
