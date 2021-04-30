import numpy as np

from module import get_input_optimizer, get_style_model_and_losses


def style_transfer(
    cnn,
    normalization_mean,
    normalization_std,
    content_img,
    style_img,
    input_img,
    num_steps=300,
    style_weight=1000000,
    content_weight=1,
    verbose=False,
    use_resnet=False,
):

    history = []
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img
    )
    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score

            # Record history
            if use_resnet:
                history.append((run[0], style_score, content_score))
            else:
                history.append((run[0], style_score.item(), content_score.item()))
            loss.backward()

            run[0] += 1
            if verbose:
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    if use_resnet:
                        print(
                            "Style Loss : {:4f} Content Loss: {:4f}".format(
                                style_score, content_score
                            )
                        )
                    else:
                        print(
                            "Style Loss : {:4f} Content Loss: {:4f}".format(
                                style_score.item(), content_score.item()
                            )
                        )
                    print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img, np.stack(history).T.tolist()
