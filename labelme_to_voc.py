import json
from pascal_voc_writer import Writer
from pathlib import Path
from typing import Iterable
import shutil


class Shape:

    def __init__(self, shape_dict):
        self._shape_dict = shape_dict

    @property
    def label(self):
        return self._shape_dict["label"]

    @property
    def points(self):
        return self._shape_dict['points']

    @property
    def points_decoded(self):
        (xmin, ymin), (xmax, ymax) = self.points
        return tuple(map(int, (xmin, ymin, xmax, ymax)))

    @property
    def xmin(self):
        self._shape_dict['points'][0]

    @property
    def ymin(self):
        self._shape_dict['points'][0]


class ReadJson:

    def __init__(
            self,
            file_path: Path,
    ):
        self.anno_file_path = file_path
        self.parsed_json = self._get_parse_file_contents()

    def _get_parse_file_contents(self) -> dict:
        with self.anno_file_path.open('r') as f:
            return json.load(f)

    @property
    def anno_file_name(self):
        return self.anno_file_path.stem

    @property
    def image_name(self):
        return self.parsed_json['imagePath']

    @property
    def shapes(self):
        for shape in  self.parsed_json["shapes"]:
            yield Shape(shape)

    @property
    def image_height(self):
        return self.parsed_json["imageHeight"]

    @property
    def image_width(self):
        return self.parsed_json["imageWidth"]

    @property
    def image_height(self):
        return self.parsed_json["imageHeight"]



class Convert:
    def __init__(
            self,
            jsons: Iterable[ReadJson],
            base_image_dir: Path,
            voc_save_dir,
            copy_image_files=False
    ):
        self.jsons = jsons
        self._img_dir = base_image_dir
        self._voc_save_dir = voc_save_dir
        assert self._voc_save_dir.exists(), 'provided path does not exist.'
        self._should_copy_images = copy_image_files

    def to_pascal_xml(self):
        labels = {}
        for label_me_json in self.jsons:
            filename = self.get_image_path(label_me_json.anno_file_path).stem + '.xml'
            xml_save_path = self.anno_dump_dir.joinpath(filename)
            src = self.get_image_path(label_me_json.image_name)
            dst = self.image_dump_dir.joinpath(label_me_json.image_name)
            writer = Writer(
                str(label_me_json.image_name),
                label_me_json.image_width, label_me_json.image_height
            )
            for obj in label_me_json.shapes:
                labels[obj.label] = 1
                label = obj.label
                if len(obj.label) == 1:
                    label = "Number_plate"
                writer.addObject(label, *obj.points_decoded)
            save_path = str(xml_save_path)
            writer.save(save_path)
            if self._should_copy_images:
                shutil.copy(str(src), str(dst))
            print('saved successfully in {}'.format(save_path))
        self.write_labels(list(labels.keys()))

    def write_labels(self, label_list):
        labels_text = '\n'.join(label_list)
        self._voc_save_dir.joinpath('labels.txt').write_text(labels_text)

    def get_image_path(self, image_name):
        image_path = self._img_dir.joinpath(image_name)
        assert image_path.exists(), '{image_path} does not exist.'
        return image_path

    @property
    def anno_dump_dir(self) -> Path:
        annotation_dir = self._voc_save_dir.joinpath('Annotations')
        annotation_dir.mkdir(parents=True, exist_ok=True)
        return annotation_dir

    @property
    def image_dump_dir(self) -> Path:
        image_dump_dir = self._voc_save_dir.joinpath('JPEGImages')
        image_dump_dir.mkdir(parents=True, exist_ok=True)
        return image_dump_dir


if __name__=="__main__":
    image_anno_path = Path('/home/akarsh/temp/pytorch-ssd/data/number_plate_labels/labelme_anno/IMG_3974.json')
    image_dir = Path('/home/akarsh/temp/pytorch-ssd/data/number_plate_labels/Datumaro/dataset/images')
    voc_out_dir = Path('/home/akarsh/temp/pytorch-ssd/data/number_plate_labels/voc_trial')
    copy_images = True
    voc_out_dir.mkdir(parents=True, exist_ok=True)
    # c = ReadJson(image_anno_path)
    # parsed_json = c.parsed_json
    # shapes = iter(c.shapes)
    # shape = next(shapes)
    # assert shape.points_decoded == tuple(map(int, (
    #     1002.3333333333335,
    #     2258.5,
    #     1981.5,
    #     2508.5
    # )))
    # assert c.image_width == 3024
    # assert c.image_height == 4032

    image_annos_iterator = map(
        ReadJson,
        image_anno_path.parent.glob("*.json")
    )
    conversion = Convert(
        jsons=image_annos_iterator,
        base_image_dir=image_dir, voc_save_dir=voc_out_dir,
        copy_image_files=copy_images)
    conversion.to_pascal_xml()
# Hand raise feature...
