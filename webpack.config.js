const path = require('path');

module.exports = {
  context: path.resolve(__dirname, 'src'),
  entry: {
    index: ['./index.js'],
  },
  output: {
    path: path.join(__dirname, '/dist'),
    filename: '[name].bundle.js',
    libraryTarget: "umd"
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /(node_modules)/,
        use: {
          loader: 'babel-loader'
        }
      },
    ]
  },
  externals: {
    '@tensorflow/tfjs': '@tensorflow/tfjs',
  },
};

