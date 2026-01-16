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
#       SCIENTIFIC FIREWORK SETTINGS
# ==========================================
AUDIO_FILE = 'nightingale.mp3' 
OUTPUT_BASE = 'axis_geometry'

# --- Mode Selection ---
SHOW_PREVIEW = False    # True = Watch Live (Silent), False = Save High-Quality Video

# --- Visual Physics ---
SPARK_COUNT = 50        # Particles per firework
EXPLOSION_SIZE = 1.0    # Radius of explosion
PITCH_THRESHOLD = 0.9   # 0.0 to 1.0. Below this = Red Circle. Above = Firework.
BACKGROUND_COLOR = '#050505' # Very dark grey (better for axes visibility)

# --- Analysis Settings ---
N_MFCC = 20
HOP_LENGTH = 512
DURATION = 30
K_NEIGHBORS = 4         # Fewer neighbors looks cleaner with the infinite line
FPS = 60

def get_unique_filename(base_name):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{base_name}_{timestamp}.mp4"

def generate_axis_video(audio_path):
    print(f"\n--- INITIATING AXIS GEOMETRY ENGINE ---")
    
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
    print(f">> Weaving Mesh...")
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm='ball_tree').fit(mfcc_3d)
    _, indices = nbrs.kneighbors(mfcc_3d)

    # 5. Graphics Setup
    print(">> Setting up Scientific Canvas...")
    fig = plt.figure(figsize=(16, 9), facecolor=BACKGROUND_COLOR)
    ax = fig.add_subplot(111, projection='3d', facecolor=BACKGROUND_COLOR)
    
    # --- ENABLE AXES ---
    ax.set_axis_on()
    ax.grid(True)
    
    # Style the panes (The walls of the 3D box)
    ax.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
    ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
    ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
    
    # Style the grid lines
    grid_style = {"color": "white", "linewidth": 0.3, "alpha": 0.2}
    ax.xaxis._axinfo["grid"].update(grid_style)
    ax.yaxis._axinfo["grid"].update(grid_style)
    ax.zaxis._axinfo["grid"].update(grid_style)

    # Axis Labels
    ax.set_xlabel('Timbre X', color='white')
    ax.set_ylabel('Timbre Y', color='white')
    ax.set_zlabel('Timbre Z', color='white')
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    ax.tick_params(axis='z', colors='gray')

    # Coordinate Text Display
    coord_text = ax.text2D(0.05, 0.90, "", transform=ax.transAxes, color='cyan', fontsize=12, family='monospace')

    # Data Prep
    # Normalize pitch for the "Red vs Firework" logic
    norm_centroid = (centroid - np.min(centroid)) / (np.max(centroid) - np.min(centroid))
    rms_norm = rms / np.max(rms)

    # --- GRAPHIC ELEMENTS ---
    
    # A. The Infinite Path (Keeps track of everywhere we've been)
    # We initialize it with empty data. Matplotlib 3D plot returns a list [line]
    line_path, = ax.plot([], [], [], color='white', alpha=0.4, linewidth=0.8)

    # B. The "Red Circle" (Low Freq Mode)
    scat_red_circle = ax.scatter([0], [0], [0], c='red', s=200, alpha=0.6, marker='o', edgecolors='red')

    # C. The "Firework Sparks" (High Freq Mode)
    dummy_x = np.zeros(SPARK_COUNT)
    scat_sparks = ax.scatter(dummy_x, dummy_x, dummy_x, c='gold', marker='*', alpha=0.9, s=20)

    # D. The Core (Always present center)
    scat_core = ax.scatter([0], [0], [0], c='white', s=50, alpha=1.0)

    # Set Camera Limits
    ax.set_xlim(mfcc_3d[:,0].min(), mfcc_3d[:,0].max())
    ax.set_ylim(mfcc_3d[:,1].min(), mfcc_3d[:,1].max())
    ax.set_zlim(mfcc_3d[:,2].min(), mfcc_3d[:,2].max())

    noise_base = np.random.normal(0, 1, (len(mfcc_3d), SPARK_COUNT, 3))
    audio_frames_per_video_frame = (sr / HOP_LENGTH) / FPS
    total_video_frames = int(len(mfcc_3d) / audio_frames_per_video_frame)

    def update(frame_num):
        idx = int(frame_num * audio_frames_per_video_frame)
        # Stop if we run out of data
        if idx >= len(mfcc_3d): return line_path,

        # --- 1. INFINITE PATH LOGIC ---
        # We plot from 0 to idx (Everything so far)
        path_x = mfcc_3d[0:idx, 0]
        path_y = mfcc_3d[0:idx, 1]
        path_z = mfcc_3d[0:idx, 2]
        
        line_path.set_data(path_x, path_y)
        line_path.set_3d_properties(path_z)

        # --- 2. CURRENT POSITION ---
        curr_pos = mfcc_3d[idx]
        curr_vol = rms_norm[idx]
        curr_pitch = norm_centroid[idx]
        
        # Update coordinate text
        coord_text.set_text(f"X: {curr_pos[0]:.2f}\nY: {curr_pos[1]:.2f}\nZ: {curr_pos[2]:.2f}\nPitch: {curr_pitch:.2f}")

        # Update Core
        scat_core._offsets3d = ([curr_pos[0]], [curr_pos[1]], [curr_pos[2]])

        # --- 3. DUAL MODE VISUALIZATION ---
        
        # MODE A: High Pitch -> FIREWORK
        if curr_pitch > PITCH_THRESHOLD:
            # Hide Red Circle
            scat_red_circle.set_color("none") 
            
            # Show Sparks
            explosion_radius = curr_vol * EXPLOSION_SIZE
            frame_noise = noise_base[idx]
            
            sx = curr_pos[0] + (frame_noise[:, 0] * explosion_radius)
            sy = curr_pos[1] + (frame_noise[:, 1] * explosion_radius)
            sz = curr_pos[2] + (frame_noise[:, 2] * explosion_radius)
            
            scat_sparks._offsets3d = (sx, sy, sz)
            # Gold color for firework
            scat_sparks.set_color('gold') 
            # Fix size array
            new_sizes = np.full(SPARK_COUNT, 20 + (curr_vol * 150))
            scat_sparks.set_sizes(new_sizes)
            
        # MODE B: Low Pitch -> RED CIRCLE
        else:
            # Hide Sparks
            scat_sparks.set_color("none")
            
            # Show Red Circle
            scat_red_circle._offsets3d = ([curr_pos[0]], [curr_pos[1]], [curr_pos[2]])
            scat_red_circle.set_color('red')
            # Pulse size with volume
            circle_size = 200 + (curr_vol * 400)
            scat_red_circle.set_sizes([circle_size])

        # Rotate Camera
        ax.view_init(elev=25, azim=frame_num * 0.2)
        
        return line_path, scat_core, scat_red_circle, scat_sparks, coord_text

    # --- EXECUTION ---
    ani = FuncAnimation(fig, update, frames=total_video_frames, blit=False, interval=1000/FPS)

    if SHOW_PREVIEW:
        print(">> Live Preview Started...")
        plt.show()
    else:
        output_file = get_unique_filename(OUTPUT_BASE)
        print(f">> Rendering High-Res Axis Animation to disk...")
        print(f">> Total Frames: {total_video_frames}")
        
        plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
        
        writer = FFMpegWriter(fps=FPS, metadata=dict(artist='AxisGeometry'), bitrate=8000)
        ani.save("temp_axis.mp4", writer=writer, dpi=150)
        plt.close()

        print(">> Adding Sound...")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.call([
            ffmpeg_exe, '-y',
            '-i', "temp_axis.mp4",
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            output_file
        ])
        if os.path.exists("temp_axis.mp4"): os.remove("temp_axis.mp4")
        print(f"\nSUCCESS! File saved: {output_file}")

if __name__ == "__main__":
    generate_axis_video(AUDIO_FILE)