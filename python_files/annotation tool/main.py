import os
import tkinter as tk
from tkinter import filedialog, Toplevel, StringVar, GROOVE
import json
import nibabel as nib
import numpy as np
from PIL import Image, ImageTk
import math


def update_visibility(c_im, exposure, contrast):
    c_im = c_im + exposure

    mean = np.mean(c_im)
    c_im = (c_im - mean) * contrast + mean

    c_im = np.clip(c_im, 0, 1)

    c_im = (c_im * 255).astype(np.uint8)
    c_im = Image.fromarray(c_im)
    c_im = c_im.resize((792, 792))

    return c_im


class BoundingBoxLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CXR Differences Labeling Tool")
        self.root.geometry("1550x980")

        self.root.state('zoomed')  # Maximizes the window
        self.root.configure(background='gray26')

        # Image placeholders
        self.image1 = None
        self.image2 = None
        self.tk_image1 = None
        self.cur_tk_image1 = None
        self.tk_image2 = None
        self.cur_tk_image2 = None
        self.image1_name = None
        self.image2_name = None
        self.image1_id = None
        self.image2_id = None

        # Labeling setup
        self.current_label = "Appearance"
        self.label_colors = {"Appearance": "red", "Disappearance": "green", "Persistence": "yellow"}
        self.persistence_colors = {"Increase": "red", "Decrease": "green", "None": "yellow"}
        self.tag_options = ["Option 1", "Option 2", "Option 3", "Option 4", "Other"]
        self.default_tag = self.tag_options[0]
        self.ellipses = []  # Store ellipse label data

        # UI Elements
        self.setup_ui()

        # Mouse interaction
        self.start_x = None
        self.start_y = None
        self.dragged_item = None
        self.selected_ellipse = None
        self.rotating = False
        self.stretched_ellipse = None
        self.orig_rx = None
        self.orig_ry = None
        self.orig_stretch_x = None
        self.orig_stretch_y = None
        self.undoing = False

    def setup_ui(self):
        usage_frame = tk.Frame(self.root)
        usage_frame.pack(side=tk.TOP, pady=5)
        usage_frame.configure(background='gray26')

        self.load_button1 = tk.Button(usage_frame, text="Load Pair", command=self.load_pair, font='Helvetica 12 bold')
        self.load_button1.pack(side=tk.LEFT, padx=10, pady=5)

        self.undo_button = tk.Button(usage_frame, text="Undo Label", command=self.undo_label, font='Helvetica 12 bold')
        self.undo_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.reset_button = tk.Button(usage_frame, text="Reset Labels", command=self.reset_labels, font='Helvetica 12 bold')
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_button = tk.Button(usage_frame, text="Save Labels", command=self.save_labels, font='Helvetica 12 bold')
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.load_button = tk.Button(usage_frame, text="Load Labels", command=self.load_labels, font='Helvetica 12 bold')
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        label_frame = tk.Frame(self.root)
        label_frame.pack(side=tk.TOP, pady=5)
        label_frame.configure(background='gray26')

        for label in self.label_colors.keys():
            btn = tk.Button(label_frame, text=label, bg=self.label_colors[label], command=lambda l=label: self.set_label(l), font='Helvetica 12 bold')
            btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.pair_name_var = StringVar(value="No pair loaded")
        self.pair_label = tk.Label(self.root, textvariable=self.pair_name_var, font='Helvetica 16 bold', background='gray26', foreground='white')
        self.pair_label.pack(side=tk.TOP, pady=2)

        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack()
        canvas_frame.configure(background='gray26')

        tk.Label(canvas_frame, text="Prior", font='Helvetica 16 bold', background='gray26', foreground='white').grid(row=0, column=2)
        tk.Label(canvas_frame, text="Current", font='Helvetica 16 bold', background='gray26', foreground='white').grid(row=0, column=3)

        self.canvas1 = tk.Canvas(canvas_frame, width=792, height=792, bg="black")
        self.canvas1.grid(row=1, column=2)

        self.canvas2 = tk.Canvas(canvas_frame, width=792, height=792, bg="black")
        self.canvas2.grid(row=1, column=3)

        prior_visibility_frame = tk.Frame(canvas_frame)
        prior_visibility_frame.grid(row=1, column=0)
        prior_visibility_frame.configure(background='gray26')

        current_visibility_frame = tk.Frame(canvas_frame)
        current_visibility_frame.grid(row=1, column=4)
        current_visibility_frame.configure(background='gray26')

        tk.Label(prior_visibility_frame, text="Exposure ", font='Helvetica 11 bold', background='gray26', foreground='white').grid(row=0, column=0)
        tk.Label(prior_visibility_frame, text=" Contrast", font='Helvetica 11 bold', background='gray26', foreground='white').grid(row=0, column=1)
        tk.Label(current_visibility_frame, text="Exposure ", font='Helvetica 11 bold', background='gray26', foreground='white').grid(row=0, column=0)
        tk.Label(current_visibility_frame, text=" Contrast", font='Helvetica 11 bold', background='gray26', foreground='white').grid(row=0, column=1)

        self.exposure_slider_prior = tk.Scale(prior_visibility_frame, from_=-0.25, to=0.3, resolution=0.005, length=670,
                                              orient=tk.VERTICAL, label="",
                                              command=self.update_visibility_prior)
        self.exposure_slider_prior.set(0)
        self.exposure_slider_prior.grid(row=1, column=0)
        self.exposure_slider_prior.configure(background='gray64', font='Helvetica 10 bold')

        self.contrast_slider_prior = tk.Scale(prior_visibility_frame, from_=0.25, to=4., resolution=0.005, length=670,
                                              orient=tk.VERTICAL, label="",
                                              command=self.update_visibility_prior)
        self.contrast_slider_prior.set(1)
        self.contrast_slider_prior.grid(row=1, column=1)
        self.contrast_slider_prior.configure(background='gray64', font='Helvetica 10 bold')

        self.exposure_slider_current = tk.Scale(current_visibility_frame, from_=-0.25, to=0.3, resolution=0.005, length=670,
                                                orient=tk.VERTICAL, label="",
                                                command=self.update_visibility_current)
        self.exposure_slider_current.set(0)
        self.exposure_slider_current.grid(row=1, column=0)
        self.exposure_slider_current.configure(background='gray64', font='Helvetica 10 bold')

        self.contrast_slider_current = tk.Scale(current_visibility_frame, from_=0.25, to=4., resolution=0.005, length=670,
                                                orient=tk.VERTICAL, label="",
                                                command=self.update_visibility_current)
        self.contrast_slider_current.set(1)
        self.contrast_slider_current.grid(row=1, column=1)
        self.contrast_slider_current.configure(background='gray64', font='Helvetica 10 bold')

        self.reset_prior_visibility_button = tk.Button(prior_visibility_frame, text="Reset visibility", command=self.reset_prior_visibility, font='Helvetica 12 bold')
        self.reset_prior_visibility_button.grid(row=2, columnspan=2)

        self.reset_current_visibility_button = tk.Button(current_visibility_frame, text="Reset visibility", command=self.reset_current_visibility, font='Helvetica 12 bold')
        self.reset_current_visibility_button.grid(row=2, columnspan=2)

        self.canvas2.bind("<Button-1>", self.on_canvas_click)
        self.canvas2.bind("<B1-Motion>", self.on_drag)
        self.canvas2.bind("<ButtonRelease-1>", self.on_release)
        self.canvas2.bind("<Button-3>", self.start_rotation)
        self.canvas2.bind("<B3-Motion>", self.perform_rotation)
        self.canvas2.bind("<ButtonRelease-3>", self.finish_rotation)
        self.canvas2.bind("<Button-2>", self.start_stretch)
        self.canvas2.bind("<B2-Motion>", self.perform_stretch)
        self.canvas2.bind("<ButtonRelease-2>", self.finish_stretch)
        self.canvas2.bind("<Control-Button-1>", self.add_details)

    def load_image(self, filename):
        nii = nib.load(filename)
        data = nii.get_fdata().T
        if len(data.shape) == 3:
            data = data[:, :, data.shape[2] // 2]  # Middle slice

        data_st = (data - np.min(data)) / (np.max(data) - np.min(data))
        data = (data_st * 255).astype(np.uint8)

        img = Image.fromarray(data)
        img = img.resize((792, 792))
        return ImageTk.PhotoImage(img), data_st

    def load_pair(self):
        dirname = filedialog.askdirectory(title="Select a CXR Pair Directory")
        if dirname:
            contents = sorted([p for p in os.listdir(dirname) if p.endswith('.nii.gz')])
            assert len(contents) == 2

            filename1 = f'{dirname}/{contents[0]}'
            self.image1_name = filename1.split('/')[-1].split('.')[0]
            self.tk_image1, self.image1 = self.load_image(filename1)
            self.image1_id = self.canvas1.create_image(0, 0, anchor=tk.NW, image=self.tk_image1)

            filename2 = f'{dirname}/{contents[1]}'
            self.image2_name = filename2.split('/')[-1].split('.')[0]
            self.tk_image2, self.image2 = self.load_image(filename2)
            self.cur_tk_image2 = self.tk_image2
            self.reset_labels()
            # self.image2_id = self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.cur_tk_image2)

            self.pair_name_var.set(str.capitalize(dirname.split('/')[-1]))

            self.exposure_slider_prior.set(0)
            self.contrast_slider_prior.set(1)
            self.exposure_slider_current.set(0)
            self.contrast_slider_current.set(1)

    def set_label(self, label):
        self.current_label = label

    def reset_prior_visibility(self):
        self.exposure_slider_prior.set(0)
        self.contrast_slider_prior.set(1)

    def reset_current_visibility(self):
        self.exposure_slider_current.set(0)
        self.contrast_slider_current.set(1)

    def update_visibility_prior(self, event):
        if self.image1 is None:
            return

        cur_im = self.image1
        exposure = self.exposure_slider_prior.get()
        contrast = self.contrast_slider_prior.get()

        im = update_visibility(cur_im, exposure, contrast)

        self.cur_tk_image1 = ImageTk.PhotoImage(im)

        # self.canvas1.delete("all")
        # self.canvas1.create_image(0, 0, anchor=tk.NW, image=self.cur_tk_image1)
        self.canvas1.itemconfig(self.image1_id, image=self.cur_tk_image1)

    def update_visibility_current(self, event):
        if self.image2 is None:
            return

        cur_im = self.image2
        exposure = self.exposure_slider_current.get()
        contrast = self.contrast_slider_current.get()

        im = update_visibility(cur_im, exposure, contrast)

        self.cur_tk_image2 = ImageTk.PhotoImage(im)

        # self.canvas2.delete("all")
        # self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.cur_tk_image2)
        self.canvas2.itemconfig(self.image2_id, image=self.cur_tk_image2)

    def start_stretch(self, event):
        overlapping = self.canvas2.find_overlapping(event.x, event.y, event.x, event.y)
        for item in overlapping:
            for i, ellipse in enumerate(self.ellipses):
                if ellipse['id'] == item:
                    self.stretched_ellipse = ellipse
                    self.orig_rx = self.stretched_ellipse['rx']
                    self.orig_ry = self.stretched_ellipse['ry']
                    self.orig_stretch_x = event.x
                    self.orig_stretch_y = event.y
                    return

    def perform_stretch(self, event):
        if self.stretched_ellipse:
            self.stretched_ellipse['rx'] = abs(event.x - self.orig_stretch_x + self.orig_rx)
            self.stretched_ellipse['ry'] = abs(event.y - self.orig_stretch_y + self.orig_ry)

            self.canvas2.delete(self.stretched_ellipse['id'])
            new_id = self.draw_rotated_ellipse(self.stretched_ellipse['cx'], self.stretched_ellipse['cy'], self.stretched_ellipse['rx'], self.stretched_ellipse['ry'], self.stretched_ellipse['angle'], self.stretched_ellipse,
                                               outline=self.label_colors[self.stretched_ellipse['label']], width=2.5, fill="")
            self.stretched_ellipse["id"] = new_id

    def finish_stretch(self, event):
        self.stretched_ellipse = None

    def on_drag(self, event):
        if self.dragged_item:
            self.dragged_item['cx'] = event.x
            self.dragged_item['cy'] = event.y

            self.canvas2.delete(self.dragged_item['id'])
            new_id = self.draw_rotated_ellipse(self.dragged_item['cx'], self.dragged_item['cy'], self.dragged_item['rx'], self.dragged_item['ry'], self.dragged_item['angle'], self.dragged_item, outline=self.label_colors[self.dragged_item['label']],
                                               width=2.5, fill="")
            self.dragged_item["id"] = new_id

    def on_release(self, event):
        self.dragged_item = None

    def draw_rotated_ellipse(self, cx, cy, rx, ry, angle_deg, item=None, **kwargs):
        num_points = 60
        angle_rad = math.radians(angle_deg)
        points = []
        for i in range(num_points):
            t = 2 * math.pi * i / num_points
            x = rx * math.cos(t)
            y = ry * math.sin(t)
            x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
            y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)
            points.append(cx + x_rot)
            points.append(cy + y_rot)

        if item is not None and item['label'] == 'Persistence':
            self.draw_indicator_circles(item)

        return self.canvas2.create_polygon(points, **kwargs)

    def on_canvas_click(self, event):
        overlapping = self.canvas2.find_overlapping(event.x, event.y, event.x, event.y)
        for item in overlapping:
            for i, ellipse in enumerate(self.ellipses):
                if ellipse['id'] == item:
                    if self.undoing:
                        self.undoing = False
                        if ellipse.get('c1'):
                            self.canvas2.delete(ellipse['c1'])
                            self.canvas2.delete(ellipse['c2'])
                        self.ellipses.remove(ellipse)
                        self.canvas2.delete(item)
                    else:
                        self.dragged_item = ellipse
                    return

        if self.start_x is None:
            self.start_x, self.start_y = event.x, event.y
        else:
            end_x, end_y = event.x, event.y
            cx = (self.start_x + end_x) / 2
            cy = (self.start_y + end_y) / 2
            rx = abs(end_x - self.start_x) / 2
            ry = abs(end_y - self.start_y) / 2
            angle = 0
            color = self.label_colors[self.current_label]

            if self.current_label == "Persistence":
                item = self.draw_rotated_ellipse(cx, cy, rx, ry, angle, outline=color, width=2.5, fill="")
                self.ellipses.append({"cx": cx, "cy": cy, "rx": rx, "ry": ry, "angle": angle, "label": self.current_label, "id": item, "size_change": "None", "intensity_change": "None", "comment": "", "tag": self.default_tag, "tag_other": ""})
                # self.ask_persistence_details(cx, cy, rx, ry, angle, item)
            else:
                item = self.draw_rotated_ellipse(cx, cy, rx, ry, angle, outline=color, width=2.5, fill="")
                self.ellipses.append({"cx": cx, "cy": cy, "rx": rx, "ry": ry, "angle": angle, "label": self.current_label, "id": item, "comment": "", "tag": self.default_tag, "tag_other": ""})
                # self.ask_comment()

            self.start_x, self.start_y = None, None

    def ask_comment(self, ellipse):
        popup = Toplevel(self.root)
        popup.title("Label Details")
        popup.geometry("350x320")

        tag_var = StringVar(value=ellipse.get('tag', self.default_tag))
        other_var = StringVar(value=ellipse.get('tag_other', ""))

        tk.Label(popup, text="Tag:", font='Helvetica 16 bold').pack(pady=(10, 5))
        tag_menu = tk.OptionMenu(popup, tag_var, *self.tag_options)
        tag_menu.config(font='Helvetica 12')
        tag_menu.pack()

        other_label = tk.Label(popup, text="Other (free text):", font='Helvetica 12 bold')
        other_label.pack(pady=(10, 5))
        other_entry = tk.Entry(popup, textvariable=other_var, width=40)
        other_entry.pack()

        def update_other_state(*_):
            if tag_var.get() == "Other":
                other_entry.config(state=tk.NORMAL)
            else:
                other_entry.config(state=tk.DISABLED)

        tag_var.trace_add("write", update_other_state)
        update_other_state()

        tk.Label(popup, text="Comment:", font='Helvetica 16 bold').pack(pady=(15, 10))
        text_box = tk.Text(popup, wrap=tk.WORD, height=3, width=35)
        text_box.pack()

        tk.Button(
            popup,
            text="Confirm",
            font='Helvetica 14 bold',
            command=lambda: self.on_confirm_comment(popup, text_box, ellipse, tag_var.get(), other_var.get()),
        ).pack(pady=15)

    def on_confirm_comment(self, popup, text_box, ellipse, tag, tag_other):
        comment = text_box.get("1.0", tk.END).strip()
        # self.ellipses[-1]['comment'] = comment
        popup.destroy()
        ellipse['comment'] = comment
        ellipse['tag'] = tag
        ellipse['tag_other'] = tag_other.strip() if tag == "Other" else ""

    def add_details(self, event):
        overlapping = self.canvas2.find_overlapping(event.x, event.y, event.x, event.y)
        for item in overlapping:
            for i, ellipse in enumerate(self.ellipses):
                if ellipse['id'] == item:
                    if ellipse['label'] == "Persistence":
                        self.ask_persistence_details(ellipse)
                    else:
                        self.ask_comment(ellipse)
                    return

    def ask_persistence_details(self, ellipse):
        popup = Toplevel(self.root)
        popup.title("Label Details")
        popup.geometry("350x620")

        size_var = StringVar(value="None")
        intensity_var = StringVar(value="None")

        tag_var = StringVar(value=ellipse.get('tag', self.default_tag))
        other_var = StringVar(value=ellipse.get('tag_other', ""))

        tk.Label(popup, text="Size Change:", font='Helvetica 16 bold').pack(pady=10)
        tk.Radiobutton(popup, text="Increase", variable=size_var, value="Increase", font='Helvetica 12').pack()
        tk.Radiobutton(popup, text="None", variable=size_var, value="None", font='Helvetica 12').pack()
        tk.Radiobutton(popup, text="Decrease", variable=size_var, value="Decrease", font='Helvetica 12').pack()

        tk.Label(popup, text="Intensity Change:", font='Helvetica 16 bold').pack(pady=10)
        tk.Radiobutton(popup, text="Increase", variable=intensity_var, value="Increase", font='Helvetica 12').pack()
        tk.Radiobutton(popup, text="None", variable=intensity_var, value="None", font='Helvetica 12').pack()
        tk.Radiobutton(popup, text="Decrease", variable=intensity_var, value="Decrease", font='Helvetica 12').pack()

        tk.Label(popup, text="Tag:", font='Helvetica 16 bold').pack(pady=(15, 5))
        tag_menu = tk.OptionMenu(popup, tag_var, *self.tag_options)
        tag_menu.config(font='Helvetica 12')
        tag_menu.pack()

        other_label = tk.Label(popup, text="Other (free text):", font='Helvetica 12 bold')
        other_label.pack(pady=(10, 5))
        other_entry = tk.Entry(popup, textvariable=other_var, width=40)
        other_entry.pack()

        def update_other_state(*_):
            if tag_var.get() == "Other":
                other_entry.config(state=tk.NORMAL)
            else:
                other_entry.config(state=tk.DISABLED)

        tag_var.trace_add("write", update_other_state)
        update_other_state()

        tk.Label(popup, text="Comment:", font='Helvetica 16 bold').pack(pady=10)
        text_box = tk.Text(popup, wrap=tk.WORD, height=3, width=35)
        text_box.pack()

        tk.Button(
            popup,
            text="Confirm",
            font='Helvetica 14 bold',
            command=lambda: self.save_persistence_ellipse(popup, text_box, size_var.get(), intensity_var.get(), ellipse, tag_var.get(), other_var.get()),
        ).pack(pady=15)

    def save_persistence_ellipse(self, popup, text_box, size_change, intensity_change, ellipse, tag, tag_other):
        comment = text_box.get("1.0", tk.END).strip()
        popup.destroy()
        # color = self.label_colors["Persistence"]
        # item = self.draw_rotated_ellipse(cx, cy, rx, ry, angle, outline=color, width=2.5, fill="")
        # self.ellipses.append({"cx": cx, "cy": cy, "rx": rx, "ry": ry, "angle": angle, "label": "Persistence", "size_change": size_change, "intensity_change": intensity_change, "id": item, "comment": comment})
        ellipse['comment'] = comment
        ellipse['size_change'] = size_change
        ellipse['intensity_change'] = intensity_change
        ellipse['tag'] = tag
        ellipse['tag_other'] = tag_other.strip() if tag == "Other" else ""
        self.draw_indicator_circles(ellipse, init=True)

    def draw_indicator_circles(self, ellipse, init=False):
        if ellipse.get('c1'):
            self.canvas2.delete(ellipse['c1'])
            self.canvas2.delete(ellipse['c2'])
        else:
            if not init:
                return

        space = 8
        r = 6
        c1 = self.canvas2.create_oval(ellipse['cx'] - space - r, ellipse['cy'] - r, ellipse['cx'] - space + r, ellipse['cy'] + r, fill=self.persistence_colors[ellipse['size_change']])
        c2 = self.canvas2.create_oval(ellipse['cx'] + space - r, ellipse['cy'] - r, ellipse['cx'] + space + r, ellipse['cy'] + r, fill=self.persistence_colors[ellipse['intensity_change']])
        ellipse['c1'] = c1
        ellipse['c2'] = c2

    def start_rotation(self, event):
        # for ellipse in reversed(self.ellipses):
        #     dist = math.hypot(event.x - ellipse["cx"], event.y - ellipse["cy"])
        #     if dist < 25:
        overlapping = self.canvas2.find_overlapping(event.x, event.y, event.x, event.y)
        for item in overlapping:
            for i, ellipse in enumerate(self.ellipses):
                if ellipse['id'] == item:
                    self.selected_ellipse = ellipse
                    self.rotating = True
                    self.rotation_start_angle = math.degrees(math.atan2(event.y - ellipse["cy"], event.x - ellipse["cx"]))
                    self.original_angle = ellipse["angle"]
                    return

    def perform_rotation(self, event):
        if self.rotating and self.selected_ellipse:
            cx, cy = self.selected_ellipse["cx"], self.selected_ellipse["cy"]
            current_angle = math.degrees(math.atan2(event.y - cy, event.x - cx))
            delta_angle = current_angle - self.rotation_start_angle
            new_angle = (self.original_angle + delta_angle) % 360
            self.canvas2.delete(self.selected_ellipse["id"])
            new_id = self.draw_rotated_ellipse(cx, cy, self.selected_ellipse["rx"], self.selected_ellipse["ry"], new_angle, self.selected_ellipse, outline=self.label_colors[self.selected_ellipse["label"]], width=2.5, fill="")
            self.selected_ellipse["id"] = new_id
            self.selected_ellipse["angle"] = new_angle

    def finish_rotation(self, event):
        self.rotating = False
        self.selected_ellipse = None

    def reset_labels(self):
        self.canvas2.delete("all")
        # if redraw_image:
        self.image2_id = self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.cur_tk_image2)
        self.ellipses.clear()

    def undo_label(self):
        self.undoing = True
        # if self.ellipses:
        #     last = self.ellipses.pop()
        #     if last.get('c1'):
        #         self.canvas2.delete(last['c1'])
        #         self.canvas2.delete(last['c2'])
        #     self.canvas2.delete(last["id"])

    def save_labels(self):
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if filename:
            to_save = []
            for ell in self.ellipses:
                payload = {k: v for k, v in ell.items() if k not in {"id", "c1", "c2"}}
                payload.setdefault("tag", self.default_tag)
                payload.setdefault("tag_other", "")
                if payload.get("tag") != "Other":
                    payload["tag_other"] = ""
                to_save.append(payload)
            to_save.insert(0, self.image1_name + ' | ' + self.image2_name)
            with open(filename, "w") as f:
                json.dump(to_save, f, indent=4)

    def load_labels(self):
        filename = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if filename:
            self.reset_labels()
            with open(filename) as f:
                labels = json.load(f)
                for l in labels[1:]:
                    l.setdefault("tag", self.default_tag)
                    l.setdefault("tag_other", "")
                    if l.get("tag") != "Other":
                        l["tag_other"] = ""
                    item = self.draw_rotated_ellipse(l['cx'], l['cy'], l['rx'], l['ry'], l['angle'], outline=self.label_colors[l['label']], width=2.5, fill="")
                    l['id'] = item
                    self.ellipses.append(l)
                    if l['label'] == 'Persistence':
                        self.draw_indicator_circles(l, init=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = BoundingBoxLabelingApp(root)
    root.mainloop()
