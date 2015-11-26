import cPickle
import matplotlib
import matplotlib.pyplot as plt

fovea_n = [1, 5]

infile = open('fovea-size.pkl', 'rb')
(fovea_fractions, foveal_times, foveal_unweighted_cost, foveal_weighted_cost) = cPickle.load(infile)
infile.close()

matplotlib.rcParams.update({'font.size': 14})

fig = plt.figure()
ax = plt.subplot(311)

plt.plot(fovea_fractions, foveal_unweighted_cost.mean(-1), '--')
plt.legend(['centred'] + ['%d foveas' % i for i in fovea_n], loc='best', fontsize=14)
plt.xticks([])
plt.locator_params(axis='y', nbins=4)
plt.ylabel('Unweighted error')

ax = plt.subplot(312)
plt.plot(fovea_fractions, foveal_weighted_cost.mean(-1), '-')
plt.legend(['centred'] + ['%d foveas' % i for i in fovea_n], loc='best', fontsize=14)
plt.xticks([])
ax.set_ylim([3.4, 4.4])
plt.locator_params(axis='y', nbins=4)
plt.ylabel('Weighted error')

plt.subplot(313)
plt.plot(fovea_fractions, foveal_times.mean(-1).mean(-1))
plt.xlabel('Fovea size (fraction of image size)')
plt.locator_params(axis='y', nbins=4)
plt.ylabel('Run time per frame (s)')

plt.show(block=True)
