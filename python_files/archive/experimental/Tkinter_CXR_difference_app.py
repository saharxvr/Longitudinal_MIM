import os.path
import tkinter as tk
from tkinter import filedialog, messagebox
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import colors
import torchvision.transforms.v2 as v2
from models import LongitudinalMIMModelBig
import torch
from augmentations import *


@torch.no_grad()
def Monte_Carlo_pred(bl, bl_seg, fu, fu_seg):
    orig_bl = bl.clone()
    orig_fu = fu.clone()
    orig_bl_seg = bl_seg.clone()
    orig_fu_seg = fu_seg.clone()

    outputs = []

    for j in range(40):
        c_bl = random_intensity_tf(rescale_tf(random_geometric_tf(torch.cat([orig_bl, orig_bl_seg], dim=0))[0]))
        c_fu = random_intensity_tf(rescale_tf(mask_crop_tf(torch.cat([orig_fu, orig_fu_seg], dim=0))[0]))

        c_bl = c_bl.unsqueeze(0).cuda()
        c_fu = c_fu.unsqueeze(0).cuda()

        outs = model(c_bl, c_fu).squeeze().cpu()
        outs[torch.logical_or(outs > 0.9, outs < -0.9)] = 0.
        outputs.append(outs[None, ...])
        continue

    outputs = torch.cat(outputs)
    outs_mean = torch.mean(outputs, dim=0)

    return outs_mean


class NiftiSummationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ICU CXR difference finder")

        self.prior_path = None
        self.prior_data = None

        self.prior_seg_path = None
        self.prior_seg_data = None

        self.current_path = None
        self.current_data = None

        self.current_seg_path = None
        self.current_seg_data = None

        self.diff_map = None

        self.global_alpha_mult = 1.
        self.diff_val_focus = 1.

        # Buttons to select files
        self.btn_file1 = tk.Button(root, text="Select Prior NIfTI File", command=self.load_file1)
        self.btn_file1.pack(pady=5)

        self.btn_file2 = tk.Button(root, text="Select Current NIfTI File", command=self.load_file2)
        self.btn_file2.pack(pady=5)

        # Sum button
        self.btn_diff = tk.Button(root, text="Find differences", command=self.find_differences)
        self.btn_diff.pack(pady=5)

        # Global Alpha slider
        self.global_alpha_slider = tk.Scale(root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Global Alpha Value", command=self.update_global_alpha_mult, length=300)
        self.global_alpha_slider.set(self.global_alpha_mult)
        self.global_alpha_slider.pack(pady=5)

        # Value focus slider
        self.val_focus_slider = tk.Scale(root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Value focus", command=self.update_val_focus, length=300)
        self.val_focus_slider.set(self.diff_val_focus)
        self.val_focus_slider.pack(pady=5)

        # Show intensifications checkbox
        self.show_intensification = tk.BooleanVar(value=True)  # Checkbox variable
        self.chkbox_intensification = tk.Checkbutton(root, text="Show intensifications", variable=self.show_intensification, command=self.display_diff)
        self.chkbox_intensification.pack(pady=5)

        # Show subsidences checkbox
        self.show_subsidence = tk.BooleanVar(value=True)  # Checkbox variable
        self.chkbox_subsidence = tk.Checkbutton(root, text="Show subsidences", variable=self.show_subsidence, command=self.display_diff)
        self.chkbox_subsidence.pack(pady=5)

        # Frame for image canvases
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack()

        self.canvas1 = None
        self.canvas2 = None
        self.canvas_diff = None

    def update_global_alpha_mult(self, val):
        self.global_alpha_mult = float(val)
        self.display_diff()

    def update_val_focus(self, val):
        self.diff_val_focus = float(val)
        self.display_diff()

    def load_file1(self):
        self.prior_path = filedialog.askopenfilename(filetypes=[("NIfTI Files", "*.nii.gz")])
        if self.prior_path:
            # messagebox.showinfo("File Selected", f"First file: {self.file1_path}")
            self.prior_data = torch.tensor(nib.load(self.prior_path).get_fdata().T[None, ...])
            if self.prior_data.shape[-1] != 768:
                self.prior_data = resize_tf(self.prior_data)

            prior_seg_name = f'{os.path.basename(self.prior_path).split(".")[0]}_seg.nii.gz'
            self.prior_seg_path = f'{os.path.dirname(self.prior_path)}_segs/{prior_seg_name}'
            self.prior_seg_data = torch.tensor(nib.load(self.prior_seg_path).get_fdata().T[None, ...])
            if self.prior_seg_data.shape[-1] != 768:
                self.prior_seg_data = resize_tf(self.prior_seg_data)
            # self.prior_seg_data = self.prior_seg_data.view_as(self.prior_data)

            self.prior_data = rescale_tf(mask_crop_tf(torch.cat([self.prior_data, self.prior_seg_data], dim=0))[0])

            self.display_image(self.prior_data, position=1)

    def load_file2(self):
        self.current_path = filedialog.askopenfilename(filetypes=[("NIfTI Files", "*.nii.gz")])
        if self.current_path:
            # messagebox.showinfo("File Selected", f"Second file: {self.file2_path}")
            self.current_data = torch.tensor(nib.load(self.current_path).get_fdata().T[None, ...])
            if self.current_data.shape[-1] != 768:
                self.current_data = resize_tf(self.current_data)

            current_seg_name = f'{os.path.basename(self.current_path).split(".")[0]}_seg.nii.gz'
            self.current_seg_path = f'{os.path.dirname(self.current_path)}_segs/{current_seg_name}'
            self.current_seg_data = torch.tensor(nib.load(self.current_seg_path).get_fdata().T[None, ...])
            if self.current_seg_data.shape[-1] != 768:
                self.current_seg_data = resize_tf(self.current_seg_data)
            # self.current_seg_data = self.current_seg_data.view_as(self.current_data)

            self.current_data = rescale_tf(mask_crop_tf(torch.cat([self.current_data, self.current_seg_data], dim=0))[0])

            self.display_image(self.current_data, position=2)

    def find_differences(self):
        if not self.prior_path or not self.current_path:
            # messagebox.showerror("Error", "Please select two NIfTI files.")
            return

        # Check if images have the same shape
        if self.prior_data.shape != self.current_data.shape:
            messagebox.showerror("Error", "Images must have the same dimensions.")
            return

        self.diff_map = Monte_Carlo_pred(self.prior_data, self.prior_seg_data, self.current_data, self.current_seg_data)

        self.diff_val_focus = torch.max(torch.abs(self.diff_map)).item()
        self.val_focus_slider.set(self.diff_val_focus)

        # Display summed image
        self.display_diff()

    def display_image(self, image_path_or_data, position):
        if isinstance(image_path_or_data, str):
            image_data = nib.load(image_path_or_data).get_fdata().T
        else:
            image_data = image_path_or_data

        if len(image_data.shape) > 2:
            slice_data = image_data[0]
        else:
            slice_data = image_data

        fig, ax = plt.subplots()
        ax.imshow(slice_data, cmap="gray")
        ax.axis("off")

        if position == 1:
            if self.canvas1:
                self.canvas1.get_tk_widget().destroy()
            self.canvas1 = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            self.canvas1.get_tk_widget().pack(side=tk.LEFT, padx=5)
        elif position == 2:
            if self.canvas2:
                self.canvas2.get_tk_widget().destroy()
            self.canvas2 = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            self.canvas2.get_tk_widget().pack(side=tk.LEFT, padx=5)

        fig.canvas.draw()

        plt.close(fig)

    def display_diff(self):
        if not self.diff_map:
            messagebox.showerror("Error", "A difference map has not been calculated yet.")
            return

        fig, ax = plt.subplots()
        ax.imshow(self.current_data.squeeze().cpu(), cmap='gray')
        divnorm = colors.TwoSlopeNorm(vmin=min(torch.min(self.diff_map).item(), -0.01), vcenter=0., vmax=max(torch.max(self.diff_map).item(), 0.01))

        alphas_map = self.generate_alpha_map()

        imm1 = ax.imshow(self.diff_map.squeeze().cpu(), alpha=alphas_map, cmap=differential_grad, norm=divnorm)
        cbar1 = plt.colorbar(imm1, fraction=0.05, pad=0.04, ax=ax)
        ax.axis("off")

        if self.canvas_diff:
            self.canvas_diff.get_tk_widget().destroy()
        self.canvas_diff = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas_diff.get_tk_widget().pack(side=tk.LEFT, padx=5)

        fig.canvas.draw()

        plt.close(fig)

    def generate_alpha_map(self):
        # x_abs = self.diff_map.abs()
        # max_val = torch.max(x_abs).item()
        # alphas_map = 1 - (x_abs - self.diff_val_focus) / max_val
        # alphas_map *= self.global_alpha_mult
        #
        # if not self.show_intensification.get():
        #     alphas_map[self.diff_map > 0] = 0.
        # if not self.show_subsidence.get():
        #     alphas_map[self.diff_map < 0] = 0.

        x_abs = self.diff_map.abs()
        max_val = 0.3
        alphas_map = torch.clip(x_abs, max=max_val) / max_val
        alphas_map *= self.global_alpha_mult

        if not self.show_intensification.get():
            alphas_map[self.diff_map > 0] = 0.
        if not self.show_subsidence.get():
            alphas_map[self.diff_map < 0] = 0.

        return alphas_map


if __name__ == "__main__":
    differential_grad = colors.LinearSegmentedColormap.from_list('my_gradient', (
        # Edit this gradient at https://eltos.github.io/gradient/#0:3CFF3D-40:00FFB3-50:FFFCFC-60:FF9A00-100:AE0000
        (0.000, (0.235, 1.000, 0.239)),
        (0.400, (0.000, 1.000, 0.702)),
        (0.500, (1.000, 0.988, 0.988)),
        (0.600, (1.000, 0.604, 0.000)),
        (1.000, (0.682, 0.000, 0.000)))
    )

    model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id31_Epoch3_Longitudinal_DeviceInvariant_DRRs_Overlay_Inpaint_MoreData_MoreEntities_NoUnrelated_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
    model = LongitudinalMIMModelBig(dec=6).cuda()
    model.load_state_dict(torch.load(model_path)['model_dict'], strict=False)
    model.train()

    resize_tf = v2.Resize((768, 768))
    random_perspective_tf = RandomAffineWithMaskTransform()
    random_bspline_tf = RandomBsplineAndSimilarityWithMaskTransform()
    random_geometric_tf = v2.RandomChoice([random_perspective_tf, random_bspline_tf], p=[0.2, 0.8])
    random_intensity_tf = RandomIntensityTransform(clahe_p=0.25, clahe_clip_limit=(0.75, 2.5), blur_p=0., jitter_p=0.35)
    rescale_tf = RescaleValuesTransform()
    clahe_tf = RandomIntensityTransform(clahe_p=0., clahe_clip_limit=2.2, blur_p=0.0, jitter_p=0.)
    mask_crop_tf = CropResizeWithMaskTransform()

    root = tk.Tk()
    app = NiftiSummationApp(root)
    root.mainloop()