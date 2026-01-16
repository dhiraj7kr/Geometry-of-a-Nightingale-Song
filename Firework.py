import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.cm as cm
import subprocess
import os
import imageio_ffmpeg
import time

# ==========================================
#      FIREWORK VISUALIZATION SETTINGS
# ==========================================
AUDIO_FILE = 'nightingale.mp3' 
OUTPUT_BASE = 'firework_geometry'

# --- Mode Selection ---
SHOW_PREVIEW = False    # Set True to watch live (silent), False to render High-Res Video

# --- Firework Physics ---
SPARK_COUNT = 40        # Particles per firework
EXPLOSION_SIZE = 0.8    # Radius of explosion
NEON_COLOR_MAP = 'hsv'  # Colors
BACKGROUND_COLOR = '#000000' 

# --- Analysis Settings ---
N_MFCC = 20
HOP_LENGTH = 512
DURATION = 30
K_NEIGHBORS = 6
FPS = 30

def get_unique_filename(base_name):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{base_name}_{timestamp}.mp4"

def generate_firework_video(audio_path):
    print(f"\n--- INITIATING FIREWORK PARTICLE ENGINE ---")
    
    # 1. Load Audio
    try:
        y, sr = librosa.load(audio_path, duration=DURATION)
    except FileNotFoundError:
        print(f"CRITICAL: '{audio_path}' not found.")
        return

    # 2. Extract Features
    print(">> Extracting Audio DNA...")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH).T
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]

    scaler = StandardScaler()
    mfcc_scaled = scaler.fit_transform(mfcc)

    # 3. PCA (3D Space)
    print(">> Calculating Trajectories...")
    pca = PCA(n_components=3)
    mfcc_3d = pca.fit_transform(mfcc_scaled)

    # 4. Connectivity Web
    print(f">> Weaving Light Mesh...")
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm='ball_tree').fit(mfcc_3d)
    _, indices = nbrs.kneighbors(mfcc_3d)

    # 5. Graphics Setup
    print(">> Igniting Particles...")
    fig = plt.figure(figsize=(16, 9), facecolor=BACKGROUND_COLOR)
    ax = fig.add_subplot(111, projection='3d', facecolor=BACKGROUND_COLOR)
    ax.set_axis_off() 
    ax.grid(False)

    ax.text2D(0.05, 0.95, f"A C O U S T I C   F I R E W O R K S", 
              transform=ax.transAxes, color='white', fontsize=14, family='monospace')

    cmap = plt.get_cmap(NEON_COLOR_MAP)
    norm = plt.Normalize(vmin=np.min(centroid), vmax=np.max(centroid))
    all_colors = cmap(norm(centroid))
    
    rms_norm = rms / np.max(rms)

    # --- GRAPHIC ELEMENTS ---
    # A. The Trail 
    scat_trail = ax.scatter([0], [0], [0], alpha=0.3, s=5, c='white')

    # B. The Sparks (Explosion Cloud) - Initialize with dummy data to set shape
    dummy_x = np.zeros(SPARK_COUNT)
    scat_sparks = ax.scatter(dummy_x, dummy_x, dummy_x, c='white', marker='*', alpha=0.8, s=20)

    # C. The Core
    scat_core = ax.scatter([0], [0], [0], c='white', s=100, alpha=1.0, edgecolors='white')

    # D. Web Lines
    lines = [ax.plot([], [], [], color='white', alpha=0.15, linewidth=0.5)[0] for _ in range(K_NEIGHBORS)]

    # Set Camera Limits
    ax.set_xlim(mfcc_3d[:,0].min(), mfcc_3d[:,0].max())
    ax.set_ylim(mfcc_3d[:,1].min(), mfcc_3d[:,1].max())
    ax.set_zlim(mfcc_3d[:,2].min(), mfcc_3d[:,2].max())

    noise_base = np.random.normal(0, 1, (len(mfcc_3d), SPARK_COUNT, 3))
    audio_frames_per_video_frame = (sr / HOP_LENGTH) / FPS
    total_video_frames = int(len(mfcc_3d) / audio_frames_per_video_frame)

    def update(frame_num):
        idx = int(frame_num * audio_frames_per_video_frame)
        if idx >= len(mfcc_3d): return scat_trail,

        # 1. Trail Logic
        tail_len = 150
        start = max(0, idx - tail_len)
        
        # Safe slicing
        cx = mfcc_3d[start:idx, 0]
        cy = mfcc_3d[start:idx, 1]
        cz = mfcc_3d[start:idx, 2]
        
        # Only update trail if we have data
        if len(cx) > 0:
            scat_trail._offsets3d = (cx, cy, cz)
            scat_trail.set_color(all_colors[start:idx])
        
        # 2. FIREWORK LOGIC
        if idx < len(mfcc_3d):
            current_vol = rms_norm[idx]
            current_pos = mfcc_3d[idx]
            current_color = all_colors[idx]

            explosion_radius = current_vol * EXPLOSION_SIZE
            frame_noise = noise_base[idx] 
            
            spark_x = current_pos[0] + (frame_noise[:, 0] * explosion_radius)
            spark_y = current_pos[1] + (frame_noise[:, 1] * explosion_radius)
            spark_z = current_pos[2] + (frame_noise[:, 2] * explosion_radius)

            scat_sparks._offsets3d = (spark_x, spark_y, spark_z)
            scat_sparks.set_color(current_color)
            
            # FIX: Ensure size is a numpy array of correct length
            new_sizes = np.full(SPARK_COUNT, 20 + (current_vol * 100))
            scat_sparks.set_sizes(new_sizes)

            # Update Core
            scat_core._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])
            scat_core.set_sizes([100 + (current_vol * 300)]) 
            scat_core.set_color('white')

            # Update Web Connections
            current_neighbors = indices[idx]
            for i, neighbor_idx in enumerate(current_neighbors):
                lx = [mfcc_3d[idx, 0], mfcc_3d[neighbor_idx, 0]]
                ly = [mfcc_3d[idx, 1], mfcc_3d[neighbor_idx, 1]]
                lz = [mfcc_3d[idx, 2], mfcc_3d[neighbor_idx, 2]]
                lines[i].set_data(lx, ly)
                lines[i].set_3d_properties(lz)
                lines[i].set_color(current_color)

        ax.view_init(elev=20, azim=frame_num * 0.2)
        return scat_trail, scat_sparks, scat_core, *lines

    # --- EXECUTION ---
    ani = FuncAnimation(fig, update, frames=total_video_frames, blit=False, interval=1000/FPS)

    if SHOW_PREVIEW:
        print(">> Live Preview Started...")
        plt.show()
    else:
        output_file = get_unique_filename(OUTPUT_BASE)
        print(f">> Rendering High-Res Firework Animation to disk...")
        print(f">> Total Frames: {total_video_frames}")
        
        # Explicit FFmpeg path
        plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
        
        writer = FFMpegWriter(fps=FPS, metadata=dict(artist='AudioFireworks'), bitrate=8000)
        ani.save("temp_firework.mp4", writer=writer, dpi=150)
        plt.close()

        print(">> Adding Sound...")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.call([
            ffmpeg_exe, '-y',
            '-i', "temp_firework.mp4",
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            output_file
        ])
        if os.path.exists("temp_firework.mp4"): os.remove("temp_firework.mp4")
        print(f"\nSUCCESS! File saved: {output_file}")

if __name__ == "__main__":
    generate_firework_video(AUDIO_FILE)