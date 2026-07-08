import torch 
from scipy.signal import find_peaks
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "CompactSIamese"))
from data.FramePrep import smooth_curve_from_values


def desc_interframe_diff(descriptor):
    """Computes interframe differences based on cosine similarity, which serves as a measure of change between consecutive frames."""
    interframe_cos_sim = torch.nn.functional.cosine_similarity(descriptor[:-1], descriptor[1:], dim=-1)
    interframe_diff = 1 - interframe_cos_sim
    return interframe_diff

def relevance_fun(descriptor):
    """Computes relevance scores for each frame based on interframe differences ensuring that scenes with significant changes are prioritized,
      while also including the first and last frames to capture the overall context of the video."""
    #integrating the interframe differences in dpp selection makes it so that frames that are more different from their neighbors are more likely to be selected, but also ensures diversity so not only selecting frames with lots of change
    nr_frames = descriptor.shape[0]
    if nr_frames <= 1:
        return torch.ones(nr_frames)
    interframe_diff = desc_interframe_diff(descriptor)
    interframe_diff_np = interframe_diff.cpu().numpy().flatten()
    diff_smooth = torch.from_numpy(smooth_curve_from_values(interframe_diff_np, window_size=5).astype("float32"))
    relevance_scores = torch.zeros(nr_frames)
    relevance_scores[0] = diff_smooth[0] #first and last frame are relevant because they capture the overall context of the video, even if there is no change at the beginning or end
    relevance_scores[nr_frames - 1] = diff_smooth[-1]
    relevance_scores[1:nr_frames - 1] = (diff_smooth[:-1] + diff_smooth[1:]) / 2
    relevance_scores_min_max = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min() + 1e-6) # normalizing 
    return relevance_scores_min_max


def dpp_frame_selection(
    frame_features,
    max_k_divider=2,
    energy_threshold=1e-7,
    min_frames=4,
):
    """Maximizes relevance and diversity through a DPP-inspired greedy selection on frame features, using interframe differences as relevance scores. 
    The gram- schmidt selection ensures diversity while the relevance scores guide the selection towards more informative frames."""

    original_device = frame_features.device
    eps = 1e-6
    num_frames = frame_features.shape[0]
    max_frames = num_frames // max_k_divider

    if num_frames <= min_frames:
        return torch.arange(num_frames, device=original_device)

    frame_features = frame_features.cpu().float()
    if frame_features.dim() == 3: # if there are multiple features per frame, we average them to get a single feature vector per frame, which is necessary for the dpp selection
        frame_features = frame_features.mean(dim=1)

    feat = frame_features / (frame_features.norm(dim=-1, keepdim=True) + eps)
    relevance = relevance_fun(frame_features)

    weighted_features = relevance[:, None] * feat #phi matrix
    feature_dim = weighted_features.shape[1]
    basis = torch.zeros((max_frames, feature_dim)) #initialize an empty basis for the selected frames

    residuals = weighted_features.clone() # residuals are the part of the features not yet covered by the selected frames
    residual_energy = (residuals * residuals).sum(dim=1)
    selected_indices = []

    for step in range(max_frames): # loop until we have selected max_frames or energy is too low
        index = torch.argmax(residual_energy) # select the frame with the most remaining energy
        current_energy = residual_energy[index] # energy is the space that is not yet covered by the selected frames so maximizing is maximizing the determinant?
        

        selected_indices.append(index.item()) # select the frame that explains the most remaining energy, which is a combination of relevance and diversity because the energy is reduced by projecting out the already selected frames, which means that frames that are similar to already selected frames will have less energy left and therefore be less likely to be selected in the next iterations

        orthogonal = residuals[index] / (torch.sqrt(current_energy) + eps) # normalize to unit length
        basis[step] = orthogonal # add the new orthogonal vector to the basis of selected frames 

        proj_coeff = residuals @ orthogonal # how much of the orthogonal component is in each residual
        #minimizing residual energy is equivalent to maximizing the determinant of the selected set, which is equivalent to maximizing diversity
        #through this we can maximize both relevance and diversity, because the relevance is encoded in the residual energy after projecting out the already selected frames
        residuals = residuals - proj_coeff[:, None] * orthogonal # remove the component in the direction of the selected frame
        residual_energy = (residuals * residuals).sum(dim=1) # energy update after projecting out
        residual_energy[index] = -float("inf") # after selecting frame is residual is made extremely small to be sure it doesnt get picked again

    if len(selected_indices) < max_frames:
        remaining_indices = set(range(num_frames)) - set(selected_indices)
        additional_indices = sorted(remaining_indices, key=lambda i: relevance[i].item(), reverse=True)[:max_frames - len(selected_indices)]
        selected_indices.extend(additional_indices)

    final_indices = torch.tensor(selected_indices)
    final_indices, _ = torch.sort(final_indices)
    return final_indices.to(original_device)






def detect_maxima_interframe_diff(descriptor, k_divider=2):
    """Detects local maxima in interframe differences to identify significant scene changes."""
    interframe_diff = desc_interframe_diff(descriptor)
    interframe_diff_np = interframe_diff.cpu().numpy().flatten()
    diff_smooth = torch.from_numpy(smooth_curve_from_values(interframe_diff_np, window_size=5).astype("float32"))

    peaks, _ = find_peaks(diff_smooth)
    num_frames = descriptor.shape[0]
    max_k = num_frames // k_divider
    if len(peaks) > max_k:
        peak_values = interframe_diff[peaks]
        top_peaks = torch.topk(peak_values, k=max_k, largest=True).indices
        peaks = peaks[top_peaks.cpu().numpy()]
    selected_indices = torch.tensor(peaks+1, device=descriptor.device)
    return selected_indices

def select_frames_by_interframe_diff(descriptor, k_divider=2, min_frames=4):
    """Selects frames based on detected local maxima in interframe differences, ensuring that significant scene changes are captured."""
    target = max(min_frames, descriptor.shape[0] // k_divider)
    if descriptor.shape[0] <= 4:
        return torch.arange(descriptor.shape[0], device=descriptor.device)
    
    descriptor = descriptor.cpu().float()
    if descriptor.dim() == 3:
        descriptor = descriptor.mean(dim=1)

    selected_indices = detect_maxima_interframe_diff(descriptor,k_divider)
    first_frame = torch.tensor([0], device=descriptor.device)
    last_frame = torch.tensor([descriptor.shape[0] - 1], device=descriptor.device)
    selected_indices = torch.unique(torch.cat((first_frame, selected_indices, last_frame)))
    if len(selected_indices) < target:
        relevance_scores = relevance_fun(descriptor)
        remaining_indices = set(range(descriptor.shape[0])) - set(selected_indices.cpu().numpy())
        additional_indices = sorted(remaining_indices, key=lambda i: relevance_scores[i].item(), reverse=True)[:target - len(selected_indices)]
        selected_indices = torch.unique(torch.cat((selected_indices, torch.tensor(additional_indices, device=descriptor.device))))

    return selected_indices

def uniform_frame_selection(num_frames,k_divider=2,min_frames=4):
    """Selects frames uniformly across the video."""
    max_k = num_frames // k_divider
    selected_indices = torch.linspace(0, num_frames - 1, steps=max_k).long()
    return selected_indices