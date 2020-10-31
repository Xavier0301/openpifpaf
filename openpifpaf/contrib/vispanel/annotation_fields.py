class AnnotationFields:
    CATEGORIES = "categories"
    CATEGORY_ID = "id"
    CATEGORY_NAME = "name"

    IMAGES = "images"
    IMAGE_ID = "id"
    IMAGE_NAME = "file_name"

    ANNOTATIONS = "annotations"

    ANNOTATION_IMAGE_ID = "image_id"
    ANNOTATION_CATEGORY_ID = "category_id"
    ANNOTATION_BBOX = "bbox"

    ANNOTATION_ID = "id"
    ANNOTATION_SEGMENTATION = "segmentation"
    ANNOTATION_ATTRIBUTES  = "attributes"
    
    ANNOTATION_FIELD_TO_DELETE = [ANNOTATION_ID, ANNOTATION_SEGMENTATION, ANNOTATION_ATTRIBUTES]

    ANNOTATION_BBOX_X = 0
    ANNOTATION_BBOX_Y = 1
    ANNOTATION_BBOX_WIDTH = 2
    ANNOTATION_BBOX_HEIGHT = 3