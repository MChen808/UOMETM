class adni_utils:
    @staticmethod
    def merge_loader(*loaders):
        for loader in loaders:
            for data in loader:
                yield data