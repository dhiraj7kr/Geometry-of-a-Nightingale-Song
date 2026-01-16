import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# ========= CONFIG ==========
AUDIO_FILE = "song.mp3"     # <-- Put your song here
OUTPUT_VIDEO = "nightingale_geometry.mp4"
SR = 22050
HOP = 512
WINDOW = 40   # frames per geometry window
# ===========================

print("Loading audio...")
y, sr = librosa.load(AUDIO_FILE, sr=SR)

print("Extracting features...")
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

features = np.vstack([
    mfcc,
    centroid,
    chroma
]).T

print("Preparing windows...")
windows = []
for i in range(0, len(features) - WINDOW, 2):
    windows.append(features[i:i+WINDOW])

print("Total frames:", len(windows))

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor("black")
fig.patch.set_facecolor("black")

def animate(frame):
    ax.clear()
    ax.set_facecolor("black")

    X = windows[frame]

    pca = PCA(n_components=3)
    Xp = pca.fit_transform(X)

    G = nx.Graph()
    for i in range(len(Xp)):
        G.add_node(i, pos=Xp[i])

    sim = cosine_similarity(X)

    for i in range(len(sim)):
        for j in range(i+1, len(sim)):
            if sim[i, j] > 0.93:
                G.add_edge(i, j)

    # Draw edges
    for i, j in G.edges():
        x = [Xp[i,0], Xp[j,0]]
        y = [Xp[i,1], Xp[j,1]]
        z = [Xp[i,2], Xp[j,2]]
        ax.plot(x, y, z, color="white", alpha=0.15)

    energy = np.linalg.norm(X, axis=1)
    colors = energy

    ax.scatter(
        Xp[:,0], Xp[:,1], Xp[:,2],
        c=colors,
        cmap="plasma",
        s=60
    )

    ax.set_title("Geometry of a Nightingale's Song", color="white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    ax.view_init(elev=30, azim=frame*2)

    return fig,

print("Rendering animation...")
ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(windows),
    interval=50
)

print("Saving video...")
ani.save(OUTPUT_VIDEO, writer="ffmpeg", fps=20)

print("Done!")
print("Saved as:", OUTPUT_VIDEO)
