import torch
import edgeiq
import numpy as np


def object_enters_0(object_id, prediction):
    print("Frame0 {}: {} enters".format(object_id, prediction.label))

def object_exits_0(object_id, prediction):
    print("Frame0 {} exits".format(prediction.label))

def object_enters_1(object_id, prediction):
    print("Frame1 {}: {} enters".format(object_id, prediction.label))

def object_exits_1(object_id, prediction):
    print("Frame1 {} exits".format(prediction.label))


def main():

    re_identifier = edgeiq.ReIdentification("alwaysai/re_id")
    re_identifier.load(engine=edgeiq.Engine.DNN)
    re_identifier.set_per_id_gallery_limit(count=100, drop_method="drop_random")
    print("Engine: {}".format(re_identifier.engine))
    print("Accelerator: {}\n".format(re_identifier.accelerator))
    print("Model:\n{}\n".format(re_identifier.model_id))
    print("Labels:\n{}\n".format(re_identifier.labels))

    # video_stream0 is the entry side stream here (The gallery is created and maintained using this stream)
    video_stream0 = edgeiq.FileVideoStream('videos/sample.mkv').start()

    # video_stream1 is the exit side stream here (The Re-Identification is performed on objects found in this stream)
    video_stream1 = edgeiq.FileVideoStream('videos/sample.mkv').start()

    detector = edgeiq.ObjectDetection('alwaysai/ssd_mobilenet_v1_coco_2018_01_28')
    detector.load(engine=edgeiq.Engine.DNN)
    print("Engine: {}".format(detector.engine))
    print("Accelerator: {}\n".format(detector.accelerator))
    print("Model:\n{}\n".format(detector.model_id))
    print("Labels:\n{}\n".format(detector.labels))

    centroid_tracker0 = edgeiq.CentroidTracker(
            min_inertia=5,
            deregister_frames=5,
            max_distance=50, enter_cb=object_enters_0,
            exit_cb=object_exits_0)

    centroid_tracker1 = edgeiq.CentroidTracker(
            min_inertia=5,
            deregister_frames=5,
            max_distance=50, enter_cb=object_enters_1,
            exit_cb=object_exits_1)

    streamer = edgeiq.Streamer().setup()

    fps = edgeiq.FPS().start()

    try:
        while True:
            frame0 = video_stream0.read()
            frame1 = video_stream1.read()

            results0 = detector.detect_objects(frame0, confidence_level=0.5)
            results_person0 = edgeiq.filter_predictions_by_label(results0.predictions, ['person'])
            tracked_people0 = centroid_tracker0.update(results_person0)

            predictions0 = []

            for _id, person in tracked_people0.items():
                person_crop = frame0[person.box.start_y:person.box.end_y, person.box.start_x:person.box.end_x,:]
                re_identifier.add_to_gallery(person_crop, _id)
                new_label = 'object {}'.format(_id)
                person.label = new_label
                predictions0.append(person.prediction)
            frame0 = edgeiq.markup_image(frame0, predictions0)

            results1 = detector.detect_objects(frame1, confidence_level=0.5)
            results_person1 = edgeiq.filter_predictions_by_label(results1.predictions, ['person'])
            tracked_people1 = centroid_tracker1.update(results_person1)

            predictions1 = []

            for _id, person in tracked_people1.items():
                person_crop = frame1[person.box.start_y:person.box.end_y, person.box.start_x:person.box.end_x,:]
                new_id = re_identifier.re_id_image(person_crop)
                print(f"RE IDENTIFIED {person} -> {new_id.predictions[0].id}")
                new_label = 'object {}'.format(new_id.predictions[0].id)
                person.label = new_label
                predictions1.append(person.prediction)
            frame1 = edgeiq.markup_image(frame1, predictions1)
            stacked_out = np.vstack([frame0, frame1])
            streamer.send_data(stacked_out, [])

            fps.update()

            if streamer.check_exit():
                break

    finally:
        fps.stop()
        video_stream0.stop()
        video_stream1.stop()
        streamer.close()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))
        print("Program Ending")

if __name__ == "__main__":
    main()
