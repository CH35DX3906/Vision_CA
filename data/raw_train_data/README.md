---
license: cc-by-sa-4.0
task_categories:
- object-detection
language:
- en
pretty_name: HaGRID Sample 30k 384p
size_categories:
- 10K<n<100K
---

This dataset contains 31,833 images from [HaGRID](https://github.com/hukenovs/hagrid) (HAnd Gesture Recognition Image Dataset) downscaled to 384p. The original dataset is 716GB and contains 552,992 1080p images. I created this sample for a tutorial so readers can use the dataset in the free tiers of Google Colab and Kaggle Notebooks.

### Original Authors:
*  [Alexander Kapitanov](https://www.linkedin.com/in/hukenovs)
* [Andrey Makhlyarchuk](https://www.linkedin.com/in/makhliarchuk)
* [Karina Kvanchiani](https://www.linkedin.com/in/kvanchiani)

### Original Dataset Links
* [GitHub](https://github.com/hukenovs/hagrid)
* [Kaggle Datasets Page](https://www.kaggle.com/datasets/kapitanov/hagrid)

### Object Classes
```text
['call',
 'no_gesture',
 'dislike',
 'fist',
 'four',
 'like',
 'mute',
 'ok',
 'one',
 'palm',
 'peace',
 'peace_inverted',
 'rock',
 'stop',
 'stop_inverted',
 'three',
 'three2',
 'two_up',
 'two_up_inverted']
```

### Annotations
* `bboxes`: `[top-left-X-position, top-left-Y-position, width, height]`
* Multiply `top-left-X-position` and `width` values by the image width and multiply `top-left-Y-position` and `height` values by the image height.
<div style="overflow-x: auto; overflow-y: auto">
<table>
  <thead>
    <tr style="text-align: right">
      <th></th>
      <th>00005c9c-3548-4a8f-9d0b-2dd4aff37fc9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bboxes</th>
      <td>[[0.23925175, 0.28595301, 0.25055143, 0.20777627]]</td>
    </tr>
    <tr>
      <th>labels</th>
      <td>[call]</td>
    </tr>
    <tr>
      <th>leading_hand</th>
      <td>right</td>
    </tr>
    <tr>
      <th>leading_conf</th>
      <td>1</td>
    </tr>
    <tr>
      <th>user_id</th>
      <td>5a389ffe1bed6660a59f4586c7d8fe2770785e5bf79b09334aa951f6f119c024</td>
    </tr>
  </tbody>
</table>
</div>
