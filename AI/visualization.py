import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_user_profile_chart(user_profile, dataset, feature_cols, pca_model, scaler, output_path=None):
    if user_profile['track_count'] == 0:
        print("Warning: Empty user profile. Cannot generate chart.")
        return None

    user_vector_pca = user_profile['feature_vector']
    user_vector_scaled = pca_model.inverse_transform(user_vector_pca)
    user_vector_original = scaler.inverse_transform(user_vector_scaled)

    user_features = pd.DataFrame([user_vector_original[0]], columns=feature_cols)

    # Normalize features to 0-100 scale for visualization
    for col in feature_cols:
        min_val = dataset[col].min()
        max_val = dataset[col].max()
        if max_val > min_val:
            user_features[col] = (user_features[col] - min_val) / (max_val - min_val) * 100
        else:
            user_features[col] = 50

    plt.figure(figsize=(10, 8))
    categories = feature_cols

    stats = user_features.iloc[0].tolist()
    stats = np.concatenate((stats, [stats[0]]))
    categories = np.concatenate((categories, [categories[0]]))

    angles = np.linspace(0, 2*np.pi, len(feature_cols), endpoint=False).tolist()
    angles += angles[:1]

    ax = plt.subplot(111, polar=True)
    ax.fill(angles, stats, color='#1DB954', alpha=0.25)
    ax.plot(angles, stats, color='#1DB954', linewidth=2, marker='o', markersize=8)

    plt.xticks(angles[:-1], categories[:-1], size=14)

    for angle, stat, category in zip(angles[:-1], stats[:-1], categories[:-1]):
        if stat > 50:
            ha = 'left' if 0 <= angle < np.pi else 'right'
            offset = 10 if 0 <= angle < np.pi else -10
            plt.annotate(f"{int(stat)}", xy=(angle, stat), xytext=(offset, 0),
                       textcoords='offset points', ha=ha, size=12)

    plt.yticks([20, 40, 60, 80], ['20', '40', '60', '80'], color="grey", size=10)
    plt.ylim(0, 100)
    plt.title("Your Music Profile", size=20, pad=20)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path

    return plt