#!/usr/bin/env python3
"""
Function for videos of eddies.
"""
import matplotlib.animation as manimation
import numpy as np
import proplot as pplt

import figures
from climopy import const


# Movie of data on some surface, to visualize eddy evolution
# * This function uses either 'aeqd' or 'aeqa' polar projection to
#   visualize eddy evolution over one hemisphere.
# * WARNING: The default 'savefig' action ***has*** to use transparent=False,
#   or weird shit will happen e.g. all boundaries and text getting redrawn/becoming
#   way thicker after the first frame!
def xyvideo(
    vname,  # video filename
    datas,  # list of xarray Datasets
    labels,  # labels
    filename='video.gif',
    frameonly=True,
    prefix=r'$\tau_{trop}=$',  # label and values
    contourf='absvor',
    plev=500,  # pressure level and variable
    levels=None,  # plot settings
    framerate=500,
    frames=20 * 4,
    jump=1,  # number of days in integration, optionally skip every nth timestep
    width=8,
    ncols=3,
    directory='videos',  # store in 'videos' subdirectory by default
    **kwargs,
):  # for format() calls
    """
    This function creates video from lat-lon-pressure data with a time dimension.
    Idea is to show how eddies look during control run.

    Todo
    ----
    Add ability to make videos to ProPlot! Maybe have all plotting functions
    permit a *third* dimension, and use that to continually update graph and
    generate video!
    """
    # Settings for contourfs of variable
    scale = 1
    coriolis = contourf == 'absvor'  # flag to add coriolis force
    if contourf == 'vor':
        cmap = 'RdBu_r'
        units = r'$10^{-5}\,s^{-1}$'
        scale = 1e5
        name = 'relative vorticity'
        levels = pplt.arange(-10, 10, 1)
        clocator = pplt.arange(-10, 10, 2)
        extend = 'both'
    elif contourf == 'absvor':
        # cmap='hclBlue_r'
        # cmap='cbWet2_r'
        # cmap='cbHot2_r'
        # cmap='ncRainbow'
        # cmap='hclPurple_r'
        cmap = 'magma'
        units = r'$10^{-5}\,s^{-1}$'
        scale = 1e5
        name = 'absolute vorticity'
        levels = pplt.arange(0, 20, 1)
        clocator = pplt.arange(0, 20, 2)
        contourf = 'vor'  # we will index data with 'vor', add coriolis later
        extend = 'both'
    elif contourf == 'v':
        cmap = 'RdBu_r'
        units = r'$m\,s^{-1}$'
        name = 'meridional wind'
        levels = pplt.arange(-20, 20, 2)
        clocator = pplt.arange(-20, 20, 4)
        extend = 'both'
    elif contourf == 't':
        cmap = 'hclBlue'
        units = 'K'
        name = 'temperature'
        if plev == 1000:
            # levels = pplt.arange(255,325,5)
            levels = pplt.arange(255, 315, 5)
        elif plev == 850:
            levels = pplt.arange(245, 305, 5)
        else:
            raise ValueError(f'Not sure which levels to pick for temp at {plev}mb.')
        clocator = 10
        extend = 'both'
    else:
        raise ValueError(f"Unknown parameter for contouring \"{contourf}\".")

    # Load requested variables and select requested pressure level
    if len(labels) != len(datas):
        raise ValueError(f"Passed {len(datas)} datasets and {len(labels)} labels.")
    datas = [data[contourf].sel(plev=plev, method='nearest') for data in datas]
    lons = [data['lon'].values for data in datas]
    lats = [data['lat'].values for data in datas]
    time = datas[0]['time'].values  # dimensions can disagree, but timing has to be same

    # Scale longitudes by climatological zonal-mean zonal wind
    # TODO: Implement this?
    # ubars = [
    #     data['u'].sel(plev=plev, method='nearest').mean(dim=('lon', 'time'))
    #     for data in datas
    # ]

    # Initialize figure, and save frame
    nrows = -(-len(datas) // ncols)  # 'ceiling' operator
    array = np.reshape(
        [(1 + i) * int(i < len(datas)) for i in range(nrows * ncols)], (nrows, ncols)
    )
    fig, axs = pplt.subplots(
        array=array,
        tight=False,
        left=0.1,
        right=0.1,
        bottom=0.1,
        top=0.45,
        wspace=0.1,
        hspace=0.1,
        width=width,
        bwidth=0.25,
        bspace=0.4,
        proj='aeqa',
        lat_0=90,
        lat_min=0,
    )  # for projection

    # Format stuff
    abc = pplt.rc.category('abc')  # dictionary for text settings
    kwargs = kwargs.copy()
    kwargs.setdefault('lonlocator', 30)
    kwargs.setdefault('latlocator', 20)
    for ax, data, label in zip(axs, datas, labels):
        ax.format(
            uctitle=prefix + str(label), **kwargs
        )  # format; override defaults with kwargs
    vspace = 0.15 / fig.height
    hspace = 0.25 / fig.width
    ax.text(
        0.5,
        1 - 1.5 * vspace,
        f'{plev:.0f}mb {name}',
        va='bottom',
        ha='center',
        transform=fig.transFigure,
        **abc,
    )
    t = ax.text(
        0.5 - hspace,
        1 - 2.5 * vspace,
        'day 0',
        va='bottom',
        ha='left',
        transform=fig.transFigure,
        **abc,
    )
    # Clever stuff to avoid drawing things
    colors = pplt.colors(cmap, N=len(levels) + 1, sample='full', interp=True)
    fig.colorbar(
        colors,
        loc='b',
        values=0.5 * (levels[1:] + levels[:-1]),
        clocator=clocator,
        clabel=units,
        extend=extend,
    )
    print('Figure ready.')

    # Function for updating contours, without redrawing stuff unnecessarily!
    # * Note, need to make sure contour mappables stored in mutable container, so
    #   can re-write to container each time we draw a frame.
    # * In this case, they are stored in list.
    # * One problem is, we might want to use 'init_func' to format axes
    #   and draw colorbars, but then
    cs = [None] * len(datas)  # always write to container; never re-define it!

    def update(itime):
        print(f'Day {float(time[itime]):.2f}.')  # i'th timestep in synoptic intervals
        # Update graphics, after removing old ones
        for i, data in enumerate(datas):
            print(f'Subplot {i:d}.')
            plot = scale * data.isel(time=itime).values.T  # tranpose to lon by lat
            if (
                coriolis
            ):  # note plot is lon by lat, and broadcasting happens right-to-left
                plot += scale * 2 * const.Omega * np.sin(lats[i] * const.pi / 180)
            if cs[i] is not None:
                for _ in cs[i].collections:
                    _.remove()  # remove each mappable object
            cs[i] = axs[i].contourf(
                lons[i], lats[i], plot, levels=levels, cmap=cmap, extend=extend
            )
        # Update text label
        text = f'day {time[itime]//1:.0f}'
        t.set_text(text)  # update the label
        if itime == 0:
            t.set_text('')
            fig.savefig(f'{directory}/{filename}_frame_0.png')
            t.set_text(text)
        elif itime == 1:
            fig.savefig(f'{directory}/{filename}_frame_1.png')

    # And now create video! The video is only actually 'drawn' when
    # you save it below; first matplotlib creates this abstract object
    # containing instructions for drawing.
    # ext, writer = 'mp4', manimation.AVConvWriter(bitrate=1600)
    ext, writer = 'gif', manimation.ImageMagickWriter(bitrate=8000)  # 8Mbps is 1080p
    mov = manimation.FuncAnimation(
        fig,
        update,
        frames=range(0, frames, jump),
        init_func=lambda: True,  # default is to use the update function!
        interval=framerate,
        blit=False,
        repeat=True,
    )
    mov.save(
        f'{directory}/{filename}.{ext}',
        dpi=200,
        writer=writer,
        savefig_kwargs={'transparent': False},
    )  # WARNING: very important here!
    return mov  # return movie object


def yzvideo(
    vname,  # video filename
    datas,  # list of xarray Datasets
    labels,  # label and values
    framerate=500,
    frames=20 * 4,
    jump=20,  # average over 5 day intervals?
    directory='videos',  # store in 'videos' subdirectory by default
    contourf='t',
    contour='u',  # stuff to plot
    **kwargs,
):  # other vars passed to xsections
    """
    This function creates video from lat-pressure data with a time dimension.
    Idea is to show spindown process.
    """
    # First simply pass everything to the "main" plot; create a figure
    abc = pplt.rc.category('abc')  # dictionary for text settings
    init = [data.isel(time=slice(0, jump)).mean(dim='time').squeeze() for data in datas]
    time = datas[0]['time'].values  # for labels and stuff
    clabel = init[0][contour].climo.long_name
    cflabel = init[0][contourf].climo.long_name
    fig, axs, _ = figures.xsections(
        init, labels, contourf=contourf, contour=contour, top=0.4, **kwargs
    )
    artists_box = [_]  # just put in a mutable container

    # Text labels
    add = fig.left / fig.width
    scale = (fig.width - fig.left - fig.right) / fig.width
    vspace = 0.15 / fig.height
    hspace = 0.25 / fig.width
    axs[0].text(
        add + scale * 0.5,
        1 - 1.5 * vspace,
        f'{cflabel[0].upper()}{cflabel[1:]} and {clabel}',
        va='bottom',
        ha='center',
        transform=fig.transFigure,
        **abc,
    )
    t = axs[0].text(
        add + scale * 0.5 - hspace,
        1 - 2.5 * vspace,
        'day 0',
        va='bottom',
        ha='left',
        transform=fig.transFigure,
        **abc,
    )

    # Prepare animations
    # See: https://stackoverflow.com/a/42398244/4970632
    # need to add new objects drawn from scratch to an already allocated
    # list/dict/whatever and continually update them
    # TODO: Add builtin proplot video support
    def update(itime):
        print(f'Day {time[itime]:.2f}.')  # i'th timestep in synoptic intervals
        if artists_box[0] is not None:
            for _ in artists_box[0]:
                _.remove()
        aves = [
            data.isel(time=slice(itime, itime + jump)).mean(dim='time').squeeze()
            for data in datas
        ]
        *_, artists = figures.xsections(
            aves, labels, fig=fig, axs=axs, contourf=contourf, contour=contour, **kwargs
        )
        artists_box[0] = artists
        # Update text
        t.set_text(f'day {time[itime]//1:.0f}')  # update the label

    # Create movie, and save
    mov = manimation.FuncAnimation(
        fig,
        update,
        frames=range(0, frames, jump),
        interval=framerate,
        blit=False,
        repeat=True,
    )
    ext, writer = 'gif', manimation.ImageMagickWriter(bitrate=1600)
    mov.save(
        f'{directory}/{vname}.{ext}',
        dpi=200,
        writer=writer,
        savefig_kwargs={'transparent': False},  # WARNING: very important here!
    )
    return mov
