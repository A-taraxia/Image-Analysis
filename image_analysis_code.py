import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.models import squeezenet1_1
import os

def reciprocal_rank_normalization(L, num_items):
    tau_i = np.arange(num_items)
    tau_j = np.random.permutation(tau_i)
    rho_n = 2 * L - (tau_i + tau_j)
    rho_n[rho_n < 0] = 0
    return rho_n

def rank_lists(T, W):
    # Define the initial ranking scores based on the given ranked list T
    ranking_scores = np.zeros(len(T))

    # Calculate the ranking scores based on the affinity matrix W
    for i, sublist in enumerate(T):
        ranking_scores[i] += W[i-1, i]  # Subtract 1 to convert from one-based to zero-based indexing

    # Sort the ranked lists based on the calculated ranking scores
    sorted_indices = np.argsort(ranking_scores)
    sorted_T = [T[i] for i in sorted_indices]

    return sorted_T

# Load the pre-trained SqueezeNet model
model = squeezenet1_1(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    image = Image.open(image_path)
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)

    return output.squeeze().cpu().numpy()

# Sample list of four images (replace this with your actual data)
# Define the image filenames
image_filenames = ['I1.jpg', 'I2.jpg', 'I3.jpg', 'I4.jpg']

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full file paths
image_list = [os.path.join(script_dir, filename) for filename in image_filenames]

# Feature extraction for each image
features = np.array([extract_features(image_path) for image_path in image_list])

# Convert feature matrix to a sparse incidence matrix
H = np.transpose(features)

# Compute similarity measure based on Cartesian product of hyperedges
num_vertices = H.shape[1]  # Determine the number of vertices from the shape of H
S = np.zeros((num_vertices, num_vertices))  # Initialize similarity matrix S

# Compute pairwise similarities using Cartesian product of hyperedges
cartesian_product = []
for i in range(num_vertices):
    for j in range(num_vertices):
        # Calculate pairwise similarity relationship p(eq, vi, vj)
        similarity = np.dot(features[:, i], features[:, j])
        S[i, j] = similarity
        # Add Cartesian Product of Hyperedge Elements to the list
        cartesian_product.append((i+1, j+1))  # +1 to convert from zero-based to one-based indexing

# Reciprocal Rank Normalization
L = 4  # Top-L positions
num_items = len(image_list)
rho_n = reciprocal_rank_normalization(L, num_items)

# Hyperedge Definition and Weight Assignment
probabilistic_incidence_matrix = np.zeros_like(S)

for i in range(num_items):
    for j in range(num_items):
        membership_measure = rho_n[i] * rho_n[j]  # Use reciprocal rank normalization as the membership measure
        probabilistic_incidence_matrix[i, j] = membership_measure

# Hyperedge Weight Calculation
hyperedge_weights_calculated = np.sum(probabilistic_incidence_matrix, axis=1)

# Define W based on C and S (element-wise multiplication)
C = probabilistic_incidence_matrix  # Placeholder for C
W = C * S  # Calculate W based on element-wise multiplication of C and S

# Define the Cartesian Product of Hyperedge Elements
cartesian_product_str = ""
for i in range(0, len(cartesian_product), len(image_filenames)):
    cartesian_product_str += "\n".join(map(str, cartesian_product[i:i+4])) + "\n"

# Display results
print('Pairwise Similarity Matrix (S):')
print(S)

print('Probabilistic Incidence Matrix:')
print(probabilistic_incidence_matrix)

print('Hyperedge Weights (Calculated):')
print(hyperedge_weights_calculated)

print('Affinity Matrix (W):')
print(W)

print('Cartesian Product of Hyperedge Elements:')
print(cartesian_product_str)

# Display the images based on the sorting result
indices = np.argsort(hyperedge_weights_calculated)[:num_items]
print('Stable Sorting Result (Indices of Multimedia Objects at Top-L Positions):')
print(indices + 1)

# Initialize ranked lists T
T = [[i+1 for i in range(num_items)]]

# Υποθετικό σύνολο δεδομένων αναφοράς
ground_truth = {
    'I1.jpg': ['I4.jpg','I3.jpg'],
    'I2.jpg': [],
    'I3.jpg': ['I4.jpg','I1.jpg'],
    'I4.jpg': ['I1.jpg', 'I3.jpg']
}

# Υποθετική εφαρμογή του αλγορίθμου στο σύνολο δεδομένων ελέγχου
results = {
    'I1.jpg': ['I2.jpg','I3.jpg', 'I4.jpg'],
    'I2.jpg': ['I3.jpg', 'I4.jpg', 'I1.jpg'],
    'I3.jpg': ['I4.jpg','I2.jpg', 'I1.jpg'],
    'I4.jpg': ['I1.jpg', 'I2.jpg','I3.jpg']
}

# Υπολογισμός ακρίβειας
total_queries = len(ground_truth)
correct_results = 0
total_relevant_images = 0
returned_relevant_images = 0

for query_image, related_images in ground_truth.items():
    if query_image in results:
        returned_images = results[query_image]
        total_relevant_images += len(related_images)
        for related_image in related_images:
            if related_image in returned_images:
                correct_results += 1
                returned_relevant_images += 1

accuracy = correct_results / total_queries
precision = returned_relevant_images / (len(results) * len(ground_truth))
recall = returned_relevant_images / total_relevant_images

print(f"Ακρίβεια: {accuracy}")
print(f"Προσέγγιση: {precision}")
print(f"Ανάκληση: {recall}")


# Add text annotations for matrices and weights
annotations = [
    f"Pairwise Similarity Matrix (S):\n{S}",
    f"Probabilistic Incidence Matrix:\n{probabilistic_incidence_matrix}",
    f"Hyperedge Weights (Calculated):\n{hyperedge_weights_calculated}",
    f"Affinity Matrix (W):\n{W}",
    f"Cartesian Product of Hyperedge Elements:\n{cartesian_product_str}"
]
 #precision data
precision_list=[
    f"Accuracy:{accuracy}\n"
    f"Precision:{precision}\n"
    f"Recall:{recall}"
]

# Perform the iterative ranking procedure
num_iterations = min(3, num_items)  # Define the number of iterations, limited by the number of items
for t in range(num_iterations):
    # Compute new set of ranked lists T(t+1) based on affinity matrix W(t)
    T.append(rank_lists(T[-1], W))  # Placeholder function, replace with your actual ranking function
    print(f"Iteration {t+1} - Ranked Lists:")
    for i, ranked_list in enumerate(T[-1]):
        print(f"Rank {i+1}: {ranked_list}")
    print()


# Calculate the number of rows needed for annotations
num_annotation_rows = len(annotations)
num_annotation_cols = 1  # One annotation column
num_cols_needed = num_items + num_annotation_cols

# Create a new subplot grid with enough rows for images and annotations
fig, axes = plt.subplots(num_annotation_rows, num_cols_needed, figsize=(15, 12))

# Display images
for i in range(num_items):
    title = f'Image {indices[i] + 1}' if i != 0 else f'Query Image: Image {indices[i] + 1}'  # Different title for the first image
    axes[0, i].imshow(np.array(Image.open(image_list[indices[i]])))
    axes[0, i].set_title(title)  # Adjust index to start from 1
    axes[0, i].axis('off')

# Display annotations
for i, annotation in enumerate(annotations):
    axes[i, num_items].text(0.5, 0.5, annotation, fontsize=8, ha='center', va='center', wrap=True)
    axes[i, num_items].axis('off')

#display precision data
for i,precision_list in enumerate(precision_list):
    axes[i+1,0].text(0.5, 0.5, precision_list, fontsize=8, ha='center', va='center', wrap=True)
    axes[i+1,0].axis('off')
    
# Adjust spacing between image columns
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# Turn off the axis completely
for ax in axes.flatten():
    ax.axis('off')

plt.show()



