
var mongoose = require('mongoose')
var Schema = mongoose.Schema;

var sentimentSchema = new Schema({
    gaze: {
        type: Number
    },
    smile: {
        type: Number
    },
    
    polarity: {
        type: Number
    },
    emotion: {
        type: Number
    },
    user_id: {
        type: mongoose.Types.ObjectId,
        ref: 'User'
    },
    video_id: {
        type: mongoose.Types.ObjectId,
        ref: 'Video'
    }
})

module.exports = mongoose.model('Sentiment', sentimentSchema);

