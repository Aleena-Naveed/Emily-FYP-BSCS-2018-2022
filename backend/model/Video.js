
var mongoose = require('mongoose')
var Schema = mongoose.Schema;

var videoSchema = new Schema({
    video_path: {
        type: String
    },
    user_id: {
        type: mongoose.Types.ObjectId,
        ref: 'User'
    }
})

module.exports = mongoose.model('Video', videoSchema);

