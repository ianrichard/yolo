def get_enabled_classes(config):
    if not config:
        return [], {}

    enabled_classes = []
    class_info = {}

    for obj_name, obj_config in config['enabled_objects'].items():
        if obj_config['enabled']:
            class_id = obj_config['class_id']
            enabled_classes.append(class_id)
            class_info[class_id] = {
                'name': obj_name,
                'color': tuple(obj_config['color'])
            }

    return enabled_classes, class_info