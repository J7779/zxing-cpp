module.exports = {
  dependencies: {
    'scandit-react-native-datacapture-core': {
      platforms: {
        android: {
          sourceDir:
            './node_modules/scandit-react-native-datacapture-core/android',
          packageImportPath:
            'import com.scandit.datacapture.reactnative.core.ScanditDataCaptureCorePackage;',
          packageInstance: 'new ScanditDataCaptureCorePackage()',
        },
      },
    },
  },
};
